import tensorflow as tf
import numpy as np
import params
import functools
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

keras = tf.keras
tfd = tfp.distributions
tf.config.experimental_run_functions_eagerly(True)

NUM_CLASSES = len(params.brand_models)


# def make_divergence_fn_for_empirical_bayes(std_prior_scale, kl_weight):
#   def divergence_fn(q, p, _):
#     log_probs = tfd.LogNormal(0., std_prior_scale).log_prob(p.stddev())
#     out = tfd.kl_divergence(q, p) - tf.math.reduce_sum(log_probs)
#     return out / kl_weight
#   return divergence_fn

# # empirical bayes for scale
# def make_prior_fn_for_empirical_bayes(init_scale_mean=-1, init_scale_std=0.1):
#   """Returns a prior function with stateful parameters for EB models."""
#   def prior_fn(dtype, shape, name, _, add_variable_fn):
#     """A prior for the variational layers."""
#     untransformed_scale = add_variable_fn(
#         name=name + '_untransformed_scale',
#         shape=shape,
#         initializer=tf.random_normal_initializer(
#             mean=init_scale_mean, stddev=init_scale_std),
#         dtype=dtype,
#         trainable=True)
#     loc = add_variable_fn(
#         name=name + '_loc',
#         initializer=keras.initializers.Zeros(),
#         shape=(1,),
#         dtype=dtype,
#         trainable=False)
#     scale = tfp.util.DeferredTensor(
#         untransformed_scale,
#         lambda x: (np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(x)))
#     dist = tfd.Normal(loc=loc, scale=scale)
#     batch_ndims = tf.size(input=dist.batch_shape_tensor())
#     return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
#   return prior_fn

# # empirical bayes for loc
# def make_prior_fn_for_empirical_bayes(init_scale_mean=-1, init_scale_std=0.1):
#   """Returns a prior function with stateful parameters for EB models."""
#   def prior_fn(dtype, shape, name, _, add_variable_fn):
#     """A prior for the variational layers."""
#     untransformed_scale = add_variable_fn(
#         name=name + '_untransformed_scale',
#         shape=(1,),
#         initializer=tf.random_normal_initializer(
#             mean=init_scale_mean, stddev=init_scale_std),
#         dtype=dtype,
#         trainable=False)
#     loc = add_variable_fn(
#         name=name + '_loc',
#         initializer=keras.initializers.Zeros(),
#         shape=shape,
#         dtype=dtype,
#         trainable=True)
#     scale = 1e-6 + tf.nn.softplus(untransformed_scale)
#     # scale = tfp.util.DeferredTensor(
#     #     untransformed_scale,
#     #     lambda x: (np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(x)))
#     dist = tfd.Normal(loc=loc, scale=scale)
#     batch_ndims = tf.size(input=dist.batch_shape_tensor())
#     return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
#   return prior_fn


class bnn(keras.Model):
    def __init__(self, kl_weight):
                #  eb_prior_fn, 
                #  examples_per_epoch):
        super(bnn, self).__init__()
        divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / 
                            tf.cast(kl_weight, dtype=tf.float32))
        # divergence_fn = make_divergence_fn_for_empirical_bayes(
        #                 params.HParams['std_prior_scale'], 
        #                 kl_weight)
        # eb_prior_fn = make_prior_fn_for_empirical_bayes()
        self.constrained_weights = None
        # no non-linearity after constrained layer
        self.constrained_conv = \
            keras.layers.Conv2D(3, (5, 5), 
                                padding='same', 
                                input_shape=[None, 
                                params.IMG_WIDTH, 
                                params.IMG_HEIGHT, 1])
        self.variational_conv1 = \
            tfp.layers.Convolution2DFlipout(
                                96, kernel_size=7,
                                strides=2, padding='same',
                                # kernel_prior_fn=eb_prior_fn,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    loc_initializer=keras.initializers.GlorotUniform(),
                                    # loc_initializer=keras.initializers.he_normal(),
                                    # untransformed_scale_initializer=tf.random_normal_initializer(
                                    #     mean=-4.0, stddev=0.1)
                                    ),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    is_singular=True,
                                    loc_initializer=keras.initializers.Zeros()),
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.variational_conv2 = \
            tfp.layers.Convolution2DFlipout(
                                64, kernel_size=5,
                                strides=1, padding='same',
                                # kernel_prior_fn=eb_prior_fn,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    loc_initializer=keras.initializers.GlorotUniform(),
                                    # loc_initializer=keras.initializers.he_normal(),
                                    # untransformed_scale_initializer=tf.random_normal_initializer(
                                    #     mean=-4.0, stddev=0.1)
                                    ),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    is_singular=True,
                                    loc_initializer=keras.initializers.Zeros()),
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.variational_conv3 = \
            tfp.layers.Convolution2DFlipout(
                                64, kernel_size=5,
                                strides=1, padding='same',
                                # kernel_prior_fn=eb_prior_fn,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    loc_initializer=keras.initializers.GlorotUniform(),
                                    # loc_initializer=keras.initializers.he_normal(),
                                    # untransformed_scale_initializer=tf.random_normal_initializer(
                                    #     mean=-4.0, stddev=0.1)
                                    ),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    is_singular=True,
                                    loc_initializer=keras.initializers.Zeros()),
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.variational_conv4 = \
            tfp.layers.Convolution2DFlipout(
                                128, kernel_size=1,
                                strides=1, padding='same',
                                # kernel_prior_fn=eb_prior_fn,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    loc_initializer=keras.initializers.GlorotUniform(),
                                    # loc_initializer=keras.initializers.he_normal(),
                                    # untransformed_scale_initializer=tf.random_normal_initializer(
                                    #     mean=-4.0, stddev=0.1)
                                    ),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    is_singular=True,
                                    loc_initializer=keras.initializers.Zeros()),
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.dense1 = tfp.layers.DenseFlipout(200,
                                # kernel_prior_fn=eb_prior_fn,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    loc_initializer=keras.initializers.GlorotUniform(),
                                    # loc_initializer=keras.initializers.he_normal(),
                                    # untransformed_scale_initializer=tf.random_normal_initializer(
                                    #     mean=-4.0, stddev=0.1)
                                    ),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    is_singular=True,
                                    loc_initializer=keras.initializers.Zeros()),
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.dense2 = tfp.layers.DenseFlipout(200,
                                # kernel_prior_fn=eb_prior_fn,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    loc_initializer=keras.initializers.GlorotUniform(),
                                    # loc_initializer=keras.initializers.he_normal(),
                                    untransformed_scale_initializer=tf.random_normal_initializer(
                                        mean=-4.0, stddev=0.1)),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    is_singular=True,
                                    loc_initializer=keras.initializers.Zeros()),
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.dense3 = tfp.layers.DenseFlipout(NUM_CLASSES,
                                # kernel_prior_fn=eb_prior_fn,
                                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    loc_initializer=keras.initializers.GlorotUniform(),
                                    # loc_initializer=keras.initializers.he_normal(),
                                    # untransformed_scale_initializer=tf.random_normal_initializer(
                                    #     mean=-4.0, stddev=0.1)
                                    ),
                                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                    is_singular=True,
                                    loc_initializer=keras.initializers.Zeros()),
                                kernel_divergence_fn=divergence_fn)

    def constrain(self):
        weights = self.constrained_conv.get_weights()[0]
        bias = self.constrained_conv.get_weights()[1]
        # check if it is converged
        if self.constrained_weights is None or np.any(self.constrained_weights!=weights):
            # Constrain the first layer
            # Scale by 10k to avoid numerical issues while normalizing
            weights = weights*10000
            # Kernel size is 5 x 5 
            # Set central values to zero to exlude them from the normalization step
            weights[2, 2, 0, :] = 0
            sumed = np.sum(weights, axis=(0,1))
            for i in range(3):
                weights[:, :, 0, i] /= sumed[0, i]
            weights[2, 2, 0, :] = -1.0

        self.constrained_conv.set_weights([weights, bias])
        self.constrained_weights = weights

    def call(self, x, training=False):
        if training:
            self.constrain()
        x = self.constrained_conv(x)

        x = self.variational_conv1(x)
        x = keras.layers.MaxPool2D(pool_size=3,
                                    strides=2,
                                    padding='SAME')(x)

        x = self.variational_conv2(x)
        x = keras.layers.MaxPool2D(pool_size=3, 
                                    strides=2)(x)

        x = self.variational_conv3(x)
        x = keras.layers.MaxPool2D(pool_size=3, 
                                    strides=2)(x)
        
        x = self.variational_conv4(x)
        x = keras.layers.MaxPool2D(pool_size=3, 
                                    strides=2)(x)

        x = keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

class vanilla(keras.Model):
    def __init__(self):
        super(vanilla, self).__init__()
        self.constrained_weights = None
        # no non-linearity after constrained layer
        self.constrained_conv = keras.layers.Conv2D(3, (5, 5), 
                                                    padding='same', 
                                                    input_shape=[None, 
                                                    params.IMG_WIDTH, 
                                                    params.IMG_HEIGHT, 1])
        self.conv1 = keras.layers.Conv2D(96, kernel_size=7,
                                         strides=2, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(64, kernel_size=5,
                                         strides=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(64, kernel_size=5,
                                         strides=1, padding='same')
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv2D(128, kernel_size=1,
                                         strides=1, padding='same')
        self.bn4 = keras.layers.BatchNormalization()
        self.dense1 = keras.layers.Dense(200)
        self.dense2 = keras.layers.Dense(200)
        self.dense3 = keras.layers.Dense(NUM_CLASSES)

    def constrain(self):
        weights = self.constrained_conv.get_weights()[0]
        bias = self.constrained_conv.get_weights()[1]
        # check if it is converged
        if self.constrained_weights is None or np.any(self.constrained_weights!=weights):
            # Constrain the first layer
            # Scale by 10k to avoid numerical issues while normalizing
            weights = weights*10000
            # Kernel size is 5 x 5 
            # Set central values to zero to exlude them from the normalization step
            weights[2, 2, 0, :] = 0
            sumed = np.sum(weights, axis=(0,1))
            for i in range(3):
                weights[:, :, 0, i] /= sumed[0, i]
            weights[2, 2, 0, :] = -1.0

        self.constrained_conv.set_weights([weights, bias])
        self.constrained_weights = weights

    def call(self, x, training=False):
        if training:
            self.constrain()

        x = self.constrained_conv(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D(pool_size=3,
                                   strides=2,
                                   padding='SAME')(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D(pool_size=3, 
                                    strides=2)(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D(pool_size=3, 
                                    strides=2)(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPool2D(pool_size=3, 
                                    strides=2)(x)

        x = keras.layers.Flatten()(x)
        x = self.dense1(x)
        x = keras.layers.Activation('relu')(x)
        x = self.dense2(x)
        x = keras.layers.Activation('relu')(x)
        x = self.dense3(x)

        return x
