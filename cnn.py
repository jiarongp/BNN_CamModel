import functools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
keras = tf.keras
tfd = tfp.distributions


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


class constrain_layer(keras.Model, keras.callbacks.Callback):
    def __init__(self, model):
        super(constrain_layer, self).__init__()
        self.model = model
        self.pre_weights = None
    # Utilized before each batch
    def on_batch_begin(self, batch, logs={}):
        # Get the weights of the first layer
        weights = self.model.layers[0].get_weights()[0]
        bias = self.model.layers[0].get_weights()[1]
        # check if it is converged
        if self.pre_weights is None or np.all(self.pre_weights != weights):
            weights = weights*10000
            weights[2, 2, :, :] = 0
            s = np.sum(weights, axis=(0,1))
            for i in range(3):
                weights[:, :, 0, i] /= s[0, i]
            weights[2, 2, :, :] = -1
            self.pre_weights = weights
            # Return the constrained weights back to the network
            self.model.layers[0].set_weights([weights, bias])


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
            initializer=keras.initializers.Zeros(),
            shape=shape,
            dtype=dtype,
            trainable=True)
        scale = 1e-6 + tf.nn.softplus(untransformed_scale)
        dist = tfd.Normal(loc=loc, scale=scale)
        batch_ndims = tf.size(input=dist.batch_shape_tensor())
        return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
    return prior_fn


def make_divergence_fn_for_empirical_bayes(std_prior_scale, examples_per_epoch):
    def divergence_fn(q, p, _):
        log_probs = tfd.LogNormal(0., std_prior_scale).log_prob(p.stddev())
        out = tfd.kl_divergence(q, p) - tf.reduce_sum(log_probs)
        return out / examples_per_epoch
    return divergence_fn


def variational_layer(inputs,
                      num_filters=16,
                      kernel_size=3,
                      strides=1,
                      activation='selu',
                      std_prior_scale=1.5,
                      eb_prior_fn=None,
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
        depth = 20
        return keras.initializers.he_normal()(shape, dtype=dtype) * depth**(-1/4)

    conv = tfp.layers.Convolution2DFlipout(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_prior_fn=eb_prior_fn,
        # kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
        #     loc_initializer=keras.initializers.he_normal()),
        kernel_divergence_fn=divergence_fn)
    
    x = inputs
    x = conv(x)
    x = keras.layers.Activation(activation)(x)
    return x


def bnn(inputs, std_prior_scale, eb_prior_fn, examples_per_epoch):

    x = keras.layers.Conv2D(3, (5, 5), padding='same')(inputs)

    x = variational_layer(inputs=x, num_filters=96, 
                          kernel_size=7, strides=2,
                          std_prior_scale=std_prior_scale,
                          eb_prior_fn=eb_prior_fn,
                          examples_per_epoch=examples_per_epoch),
    x = keras.layers.MaxPool2D(pool_size=(3,3), 
                               strides=2, 
                               padding='same')(x),
    x = variational_layer(inputs=x, num_filters=64, 
                          kernel_size=5, strides=1,
                          std_prior_scale=std_prior_scale,
                          eb_prior_fn=eb_prior_fn,
                          examples_per_epoch=examples_per_epoch)(x),
    x = keras.layers.MaxPool2D(pool_size=(3,3), 
                               strides=2, 
                               padding='same')(x),
    x = variational_layer(inputs=x, num_filters=64, 
                          kernel_size=5, strides=1,
                          std_prior_scale=std_prior_scale,
                          eb_prior_fn=eb_prior_fn,
                          examples_per_epoch=examples_per_epoch)(x),
    x = keras.layers.MaxPool2D(pool_size=(3,3), 
                               strides=2, 
                               padding='same')(x),
    x = variational_layer(inputs=x, num_filters=128, 
                          kernel_size=1, strides=1,
                          std_prior_scale=std_prior_scale,
                          eb_prior_fn=eb_prior_fn,
                          examples_per_epoch=examples_per_epoch)(x),                      
    x = keras.layers.MaxPool2D(pool_size=(3,3), 
                               strides=2, 
                               padding='same')(x),
    x = keras.layers.Flatten()(x)

    return x
