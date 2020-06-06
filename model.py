import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import params
import functools
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

NUM_CLASSES = len(params.brand_models)

def constrainLayer(weights):
    # Scale by 10k to avoid numerical issues while normalizing
    weights = weights*10000
    # Kernel size is 5 x 5 
    # Set central values to zero to exlude them from the normalization step
    weights[2, 2, :, :] = 0
    # Pass the weights 
    filter_1 = weights[:, :, 0, 0]
    filter_2 = weights[:, :, 0, 1]
    filter_3 = weights[:, :, 0, 2]
    # Normalize the weights for each filter. 
    # Sum in the 3rd dimension, which contains 25 numbers.
    filter_1 = filter_1 / np.sum(filter_1)
    filter_1[2, 2] = -1
    filter_2 = filter_2 / np.sum(filter_2)
    filter_2[2, 2] = -1
    filter_3 = filter_3 / np.sum(filter_3)
    filter_3[2, 2] = -1
    # Pass the weights back to the original matrix and return.
    weights[:, :, 0, 0] = filter_1
    weights[:, :, 0, 1] = filter_2
    weights[:, :, 0, 2] = filter_3

    return weights

# The callback to be applied at the end of each iteration. This is 
# used to constrain the layer's weights the same way Bayar and Stamm do
# at their paper.
class ConstrainLayer(tf.keras.Model, tf.keras.callbacks.Callback):
    """The callback to be applied at the end of each iteration. This is 
       used to constrain the layer's weights the same way as Bayar and Stamm do
       at their paper.
    """
    def __init__(self, model):
        super(ConstrainLayer, self).__init__()
        self.model = model
        self.tmp = None
    # Utilized before each batch
    def on_batch_begin(self, batch, logs={}):
        # Get the weights of the first layer
        all_weights = self.model.get_weights()
        weights = np.asarray(all_weights[0])
        # check if it is converged
        if self.tmp is None or np.all(self.tmp != weights):
            # Constrain the first layer
            weights = constrainLayer(weights)
            self.tmp = weights
            # Return the constrained weights back to the network
            all_weights[0] = weights
        self.model.set_weights(all_weights)

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    """Posterior function for variational layer."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1e-5)) # Calculate exp(x) - 1 for all elements in the array.
    variable_layer = tfp.layers.VariableLayer(
        2 * n,
        dtype=dtype,
        initializer=tfp.layers.BlockwiseInitializer([
            tf.keras.initializers.TruncatedNormal(mean=0., stddev=.05, seed=None),
            tf.keras.initializers.Constant(np.log(np.expm1(1e-5)))],
                                            sizes=[n, n]))

    def distribution_fn(t):
        scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, n:])
        return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                                reinterpreted_batch_ndims=1)

    distribution_layer = tfp.layers.DistributionLambda(distribution_fn)
    return tf.keras.Sequential([variable_layer, distribution_layer])

def prior_trainable(kernel_size, bias_size=0, dtype=None, num_updates=1):
    """Prior function for variational layer."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1e-5))

    def regularizer(t):
        out = tfd.LogNormal(0., 1.).log_prob(1e-5 + tf.nn.softplus(c + t[Ellipsis, -1]))
        return -tf.reduce_sum(out) / num_updates

    # Include the prior on the scale parameter as a regularizer in the loss.
    variable_layer = tfp.layers.VariableLayer(n, dtype=dtype, regularizer=regularizer)

    def distribution_fn(t):
        scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, -1])
        return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                                reinterpreted_batch_ndims=1)

    distribution_layer = tfp.layers.DistributionLambda(distribution_fn)
    return tf.keras.Sequential([variable_layer, distribution_layer])

def create_model(NUM_TRAIN_EXAMPLES):
    """ Create a Keras model using the LeNet-5 architecture.
    Returns:
        model: Compiled Keras model
    """
    # KL divergence weighted by the number of training smaoles, using lambda function 
    # to pass as input to the kernel_divergence_fn on flipout layers
    kernel_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    # Step_1: define tensorflow model
    # Here a typicall vanilla CNN is defined using three Conv layers followed 
    # by MaxPooling operations and two fully connected layers
    model = tf.keras.Sequential([
        # the first parameter defines the #-of feature maps,
        # the second parameter the filter kernel size
        tf.keras.layers.Conv2D(
            3, (5, 5), padding='same',
            input_shape=(256, 256, 1)),
        tfp.layers.Convolution2DFlipout(
            96, kernel_size=7, strides=2,
            kernel_divergence_fn=kernel_divergence_function,
            padding='same', activation=tf.nn.selu),
        tf.keras.layers.MaxPool2D(
            pool_size=(3,3), strides=2, 
            padding='same'),
        tfp.layers.Convolution2DFlipout(
            64, kernel_size=5, strides=1,
            kernel_divergence_fn=kernel_divergence_function,
            padding='same', activation=tf.nn.selu),
        tf.keras.layers.MaxPool2D(
            pool_size=(3,3), strides=2,
            padding='same'),
        tfp.layers.Convolution2DFlipout(
            64, kernel_size=5, strides=1,
            kernel_divergence_fn=kernel_divergence_function,
            padding='same', activation=tf.nn.selu),
        tf.keras.layers.MaxPool2D(
            pool_size=(3,3), strides=2,
            padding='same'),
        tfp.layers.Convolution2DFlipout(
            128, kernel_size=1, strides=1, 
            kernel_divergence_fn=kernel_divergence_function,
            padding='same', activation=tf.nn.selu),
        tf.keras.layers.MaxPool2D(
            pool_size=(3,3), strides=2,
            padding='same'),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(
            200, kernel_divergence_fn=kernel_divergence_function,
            activation=tf.nn.selu),
        tfp.layers.DenseFlipout(
            200, kernel_divergence_fn=kernel_divergence_function,
            activation=tf.nn.selu),
        tfp.layers.DenseVariational(
            units=NUM_CLASSES,
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=functools.partial(
                prior_trainable, num_updates=NUM_TRAIN_EXAMPLES),
            use_bias=True,
            kl_weight=1./NUM_TRAIN_EXAMPLES,
            kl_use_exact=True,
            name='fc1000',
            activation=tf.nn.softmax
        )
    ])
        # Model compilation
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    # Use the Categorical_Crossentropy loss since the MNIST dataset contains ten labels.
    # The Keras API will then automatically add the Kullback-Leibler divergence (contained 
    # on the individual layers of the model), to the cross entropy loss, effectively 
    # calculating the (negated) Evidence Lower Bound Loss (ELBO)
    model.compile(optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
