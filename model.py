import tensorflow as tf
import numpy as np
import params
import functools
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

tf.config.experimental_run_functions_eagerly(True)
NUM_CLASSES = len(params.brand_models)

def constrain_conv(layer, pre_weights):
    weights = layer.get_weights()[0]
    bias = layer.get_weights()[1]
    # check if it is converged
    if pre_weights is None or np.all(pre_weights != weights):
        # Constrain the first layer
        # Scale by 10k to avoid numerical issues while normalizing
        weights = weights*10000
        # Kernel size is 5 x 5 
        # Set central values to zero to exlude them from the normalization step
        weights[2, 2, :, :] = 0
        s = np.sum(weights, axis=(0,1))
        for i in range(3):
            weights[:, :, 0, i] /= s[0, i]
        weights[2, 2, :, :] = -1

    pre_weights = weights
    layer.set_weights([weights, bias])
    return pre_weights


def _posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  """Posterior function for variational layer."""
  n = kernel_size + bias_size
  c = np.log(np.expm1(1e-5))
  variable_layer = tfp.layers.VariableLayer(
      2 * n, dtype=dtype,
      initializer=tfp.layers.BlockwiseInitializer([
          tf.keras.initializers.TruncatedNormal(mean=0., stddev=.05, seed=None),
          tf.keras.initializers.Constant(np.log(np.expm1(1e-5)))], sizes=[n, n]))

  def distribution_fn(t):
    scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, n:])
    return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                           reinterpreted_batch_ndims=1)
  distribution_layer = tfp.layers.DistributionLambda(distribution_fn)
  return tf.keras.Sequential([variable_layer, distribution_layer])


def _make_prior_fn(kernel_size, bias_size=0, dtype=None):
  del dtype  # TODO(yovadia): Figure out what to do with this.
  loc = tf.zeros(kernel_size + bias_size)
  def distribution_fn(_):
    return tfd.Independent(tfd.Normal(loc=loc, scale=1),
                           reinterpreted_batch_ndims=1)
  return distribution_fn


def make_divergence_fn_for_empirical_bayes(std_prior_scale, examples_per_epoch):
    def divergence_fn(q, p, _):
        log_probs = tfd.LogNormal(0., std_prior_scale).log_prob(p.stddev())
        out = tfd.kl_divergence(q, p) - tf.reduce_sum(log_probs)
        return out / examples_per_epoch
    return divergence_fn


def make_prior_fn_for_empirical_bayes(init_scale_mean=-1, init_scale_std=0.1):
  """Returns a prior function with stateful parameters for EB models."""
  def prior_fn(dtype, shape, name, _, add_variable_fn):
    """A prior for the variational layers."""
    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=(1,),
        initializer=tf.compat.v1.initializers.random_normal(
            mean=init_scale_mean, stddev=init_scale_std),
        dtype=dtype,
        trainable=False)
    loc = add_variable_fn(
        name=name + '_loc',
        initializer=tf.keras.initializers.Zeros(),
        shape=shape,
        dtype=dtype,
        trainable=True)
    scale = 1e-6 + tf.nn.softplus(untransformed_scale)
    dist = tfd.Normal(loc=loc, scale=scale)
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
  return prior_fn

def variational_layer(inputs,
                      num_filters=16,
                      kernel_size=3,
                      strides=1,
                      activation='selu',
                      std_prior_scale=1.5,
                      eb_prior_fn=None,
                      always_on_dropout_rate=None,
                      examples_per_epoch=None):
    """2D Convolution-Batch Normalization-Activation stack builder.
    Args:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): Activation function string.
        depth (int): ResNet depth; used for initialization scale.
        variational (bool): Whether to use a variational convolutional layer.
        std_prior_scale (float): Scale for log-normal hyperprior.
        eb_prior_fn (callable): Empirical Bayes prior for use with TFP layers.
        examples_per_epoch (int): Number of examples per epoch for variational KL.
    Returns:
        x (tensor): tensor as input to the next layer
    """

    divergence_fn = make_divergence_fn_for_empirical_bayes(
        std_prior_scale, examples_per_epoch)

    def fixup_init(shape, dtype=None):
        """Fixup initialization; see https://arxiv.org/abs/1901.09321."""
        depth = 0 # what is this depth here?
        return tf.keras.initializers.he_normal()(shape, dtype=dtype) * depth**(-1/4)

    conv = tfp.layers.Convolution2DFlipout(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_prior_fn=eb_prior_fn,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            loc_initializer=tf.keras.initializers.he_normal()),
        kernel_divergence_fn=divergence_fn)
    
    x = inputs
    x = conv(x)
    x = tf.keras.layers.Activation(activation)(x) if activation is not None else x
    return x

def make_lr_scheduler(init_lr):
  """Builds a keras LearningRateScheduler."""

  def schedule_fn(epoch):
    """Learning rate schedule function."""
    rate = init_lr
    if epoch > 180:
      rate *= 0.5e-3
    elif epoch > 160:
      rate *= 1e-3
    elif epoch > 120:
      rate *= 1e-2
    elif epoch > 80:
      rate *= 1e-1

    return rate
  return tf.keras.callbacks.LearningRateScheduler(schedule_fn)

class BNN(tf.keras.Model):
    def __init__(self, 
                 std_prior_scale, 
                 eb_prior_fn, 
                 examples_per_epoch):
        super(BNN, self).__init__()
        self.constrained_weights = None
        # no non-linearity after constrained layer
        self.constrained_conv = tf.keras.layers.Conv2D(3, (5, 5), 
                                                    padding='same', 
                                                    input_shape=[None, params.IMG_WIDTH, params.IMG_HEIGHT, 1])
        self.variational_layer = functools.partial(
                                    variational_layer,
                                    activation='selu',
                                    std_prior_scale=std_prior_scale,
                                    eb_prior_fn=eb_prior_fn,
                                    examples_per_epoch=examples_per_epoch)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, 
                                                  padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tfp.layers.DenseReparameterization(
                                    NUM_CLASSES,
                                    kernel_prior_fn=eb_prior_fn,
                                    kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                        loc_initializer=tf.keras.initializers.he_normal()),
                                    kernel_divergence_fn=
                                        make_divergence_fn_for_empirical_bayes(std_prior_scale, 
                                                                               examples_per_epoch))

    def call(self, x):
        # if training:
        #     constrain_conv(self.constrained_conv, 
        #                    self.constrained_weights)
        x = self.constrained_conv(x)
        x = self.variational_layer(inputs=x, 
                                   num_filters=96, 
                                   kernel_size=7, 
                                   strides=2)
        x = self.maxpool(x)
        x = self.variational_layer(inputs=x, 
                                   num_filters=64, 
                                   kernel_size=5, 
                                   strides=1)
        x = self.maxpool(x)
        x = self.variational_layer(inputs=x, 
                                   num_filters=64, 
                                   kernel_size=5, 
                                   strides=1)
        x = self.maxpool(x)
        x = self.variational_layer(inputs=x, 
                                   num_filters=128, 
                                   kernel_size=1, 
                                   strides=1)
        x = self.maxpool(x)   
        x = self.flatten(x)
        x = self.dense(x)
        return x



# def create_model(NUM_TRAIN_EXAMPLES, variational):
#     """ Create a Keras model using the LeNet-5 architecture.
#     Returns:
#         model: Compiled Keras model
#     """

#     # KL divergence weighted by the number of training smaoles, using lambda function 
#     # to pass as input to the kernel_divergence_fn on flipout layers
#     kernel_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))

#     # Step_1: define tensorflow model
#     # Here a typicall vanilla CNN is defined using three Conv layers followed 
#     # by MaxPooling operations and two fully connected layers
#     model = tf.keras.Sequential([
#         # the first parameter defines the #-of feature maps,
#         # the second parameter the filter kernel size
#         tf.keras.layers.Conv2D(
#             3, (5, 5), padding='same'),
#         tfp.layers.Convolution2DFlipout(
#             96, kernel_size=7, strides=2,
#             kernel_divergence_fn=kernel_divergence_function,
#             padding='same', activation=tf.nn.selu),
#         tf.keras.layers.MaxPool2D(
#             pool_size=(3,3), strides=2, 
#             padding='same'),
#         tfp.layers.Convolution2DFlipout(
#             64, kernel_size=5, strides=1,
#             kernel_divergence_fn=kernel_divergence_function,
#             padding='same', activation=tf.nn.selu),
#         tf.keras.layers.MaxPool2D(
#             pool_size=(3,3), strides=2,
#             padding='same'),
#         tfp.layers.Convolution2DFlipout(
#             64, kernel_size=5, strides=1,
#             kernel_divergence_fn=kernel_divergence_function,
#             padding='same', activation=tf.nn.selu),
#         tf.keras.layers.MaxPool2D(
#             pool_size=(3,3), strides=2,
#             padding='same'),
#         tfp.layers.Convolution2DFlipout(
#             128, kernel_size=1, strides=1, 
#             kernel_divergence_fn=kernel_divergence_function,
#             padding='same', activation=tf.nn.selu),
#         tf.keras.layers.MaxPool2D(
#             pool_size=(3,3), strides=2,
#             padding='same'),
#         tf.keras.layers.Flatten(),
#         tfp.layers.DenseFlipout(
#             200, kernel_divergence_fn=kernel_divergence_function,
#             activation=tf.nn.selu),
#         tfp.layers.DenseFlipout(
#             200, kernel_divergence_fn=kernel_divergence_function,
#             activation=tf.nn.selu),
#         tfp.layers.DenseVariational(
#             units=NUM_CLASSES,
#             make_posterior_fn=posterior_mean_field,
#             make_prior_fn=functools.partial(
#                 prior_trainable, num_updates=NUM_TRAIN_EXAMPLES),
#             use_bias=True,
#             kl_weight=1./NUM_TRAIN_EXAMPLES,
#             kl_use_exact=True,
#             name='fc1000',
#             activation=tf.nn.softmax
#         )
#     ])
#     return model

# def create_model(NUM_EXAMPLES):
#     # Step_1: define tensorflow model
#     # Here a typicall vanilla CNN is defined using three Conv layers followed 
#     # by MaxPooling operations and two fully connected layers 
#     model = tf.keras.Sequential([
#         # the first parameter defines the #-of feature maps,
#         # the second parameter the filter kernel size
#         tf.keras.layers.Conv2D(3, (5, 5), padding='same'),
        
#         tf.keras.layers.Conv2D(96, (7, 7), strides=2, padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same'),

#         tf.keras.layers.Conv2D(64, (5,5), strides=1, padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2,),

#         tf.keras.layers.Conv2D(64, (5,5), strides=1, padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2,),

#         tf.keras.layers.Conv2D(128, (1,1), strides=1, padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2,),

#         tf.keras.layers.Flatten(),
        
#         tf.keras.layers.Dense(200),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dense(200),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
#     ])
    
    # optimizer = tf.keras.optimizers.Adam(lr=0.001)
    # # Use the Categorical_Crossentropy loss since the MNIST dataset contains ten labels.
    # # The Keras API will then automatically add the Kullback-Leibler divergence (contained 
    # # on the individual layers of the model), to the cross entropy loss, effectively 
    # # calculating the (negated) Evidence Lower Bound Loss (ELBO)
    # model.compile(optimizer,
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=['accuracy'])
    # return model

