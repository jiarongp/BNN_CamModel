import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
keras = tf.keras
tfd = tfp.distributions

class BaseModel(tf.keras.Model):
    def __init__(self, params):
        super(BaseModel, self).__init__()
        self.params = params
        self.log_file = self.params.log.log_file
        self.num_cls = len(self.params.dataloader.brand_models)

    def constrained_conv_update(self):
        weights = self.constrained_conv_layer.weights[0]
        for i in range(weights.shape[-1]):
            weights[2, 2, 0, i].assign(0.)
            weights[:, :, 0, i].assign(tf.math.divide(weights[:, :, 0, i],
                                    tf.math.reduce_sum(weights[:, :, 0, i])))
            weights[2, 2, 0, i].assign(-1.)
        self.constrained_conv_layer.weights[0].assign(weights)

class VanillaCNN(BaseModel):
    def __init__(self, params):
        super(VanillaCNN, self).__init__(params)
        # input_shape = (self.params.model.input_shape.width,
        #     self.params.model.input_shape.height, 1)
        
        # self.input_layer = keras.layers.Input(shape=input_shape, dtype='float32')
        self.constrained_conv_layer = \
            keras.layers.Conv2D(3, (5, 5), 
                padding='same',
                input_shape=[None, 
                    self.params.model.input_shape.width, 
                    self.params.model.input_shape.height, 
                    1])
        self.conv1 = keras.layers.Conv2D(
                        96, kernel_size=7,
                        strides=2, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(
                        64, kernel_size=5,
                        strides=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(
                        64, kernel_size=5,
                        strides=1, padding='same')
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv2D(
                        128, kernel_size=1,
                        strides=1, padding='same')
        self.bn4 = keras.layers.BatchNormalization()
        self.dense1 = keras.layers.Dense(200)
        self.dense2 = keras.layers.Dense(200)
        self.dense3 = keras.layers.Dense(self.num_cls)

    def call(self, x, training=False):
        if training:
            self.constrained_conv_update()
        x = self.constrained_conv_layer(x)
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


class EnsembleCNN(VanillaCNN):
    def __init__(self, params):
        super(EnsembleCNN, self).__init__(params)


class BayesianCNN(BaseModel):
    def __init__(self, params, kl_weight):
        super(BayesianCNN, self).__init__(params)
        self.divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / 
                                tf.cast(kl_weight, dtype=tf.float32))
        # no non-linearity after constrained layer
        self.constrained_conv_layer = \
            keras.layers.Conv2D(3, (5, 5), 
                padding='same', 
                input_shape=[None, 
                self.params.model.input_shape.width, 
                self.params.model.input_shape.height, 
                1])
        self.variational_conv1 = \
            tfp.layers.Convolution2DFlipout(
                96, kernel_size=7,
                strides=2, padding='same',
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.variational_conv2 = \
            tfp.layers.Convolution2DFlipout(
                64, kernel_size=5,
                strides=1, padding='same',
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.variational_conv3 = \
            tfp.layers.Convolution2DFlipout(
                64, kernel_size=5,
                strides=1, padding='same',
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.variational_conv4 = \
            tfp.layers.Convolution2DFlipout(
                128, kernel_size=1,
                strides=1, padding='same',
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                                activation='selu')
        self.dense1 = tfp.layers.DenseFlipout(200,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.dense2 = tfp.layers.DenseFlipout(200,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.dense3 = tfp.layers.DenseFlipout(self.num_cls,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn)

    def call(self, x, training=False):
        if training:
            self.constrained_conv_update()
        x = self.constrained_conv_layer(x)
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


# empirical bayes BayesianCNN
class EB_BayesianCNN(BayesianCNN):
    def __init__(self, params, kl_weight):
        super(EB_BayesianCNN, self).__init__(params)
        self.divergence_fn = self.make_divergence_fn_for_empirical_bayes(
                        params.HParams['std_prior_scale'], 
                        kl_weight)
        self.eb_prior_fn = self.make_prior_fn_for_empirical_bayes()
        self.constrained_conv_layer = \
            keras.layers.Conv2D(3, (5, 5), 
                padding='same', 
                input_shape=[None, 
                self.params.model.input_shape.width, 
                self.params.model.input_shape.height, 
                1])
        self.variational_conv1 = \
            tfp.layers.Convolution2DFlipout(
                96, kernel_size=7,
                strides=2, padding='same',
                kernel_prior_fn=self.eb_prior_fn,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.variational_conv2 = \
            tfp.layers.Convolution2DFlipout(
                64, kernel_size=5,
                strides=1, padding='same',
                kernel_prior_fn=self.eb_prior_fn,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.variational_conv3 = \
            tfp.layers.Convolution2DFlipout(
                64, kernel_size=5,
                strides=1, padding='same',
                kernel_prior_fn=self.eb_prior_fn,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.variational_conv4 = \
            tfp.layers.Convolution2DFlipout(
                128, kernel_size=1,
                strides=1, padding='same',
                kernel_prior_fn=self.eb_prior_fn,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                                activation='selu')
        self.dense1 = tfp.layers.DenseFlipout(200,
                kernel_prior_fn=self.eb_prior_fn,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.dense2 = tfp.layers.DenseFlipout(200,
                kernel_prior_fn=self.eb_prior_fn,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn,
                activation='selu')
        self.dense3 = tfp.layers.DenseFlipout(self.num_cls,
                kernel_prior_fn=self.eb_prior_fn,
                kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    loc_initializer=keras.initializers.GlorotUniform()),
                bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=keras.initializers.Zeros()),
                kernel_divergence_fn=self.divergence_fn)


    def make_divergence_fn_for_empirical_bayes(self, std_prior_scale, kl_weight):
        def divergence_fn(q, p, _):
            log_probs = tfd.LogNormal(0., std_prior_scale).log_prob(p.stddev())
            out = tfd.kl_divergence(q, p) - tf.math.reduce_sum(log_probs)
            return out / kl_weight
        return divergence_fn

    # empirical bayes for scale
    def make_prior_fn_for_empirical_bayes(self, init_scale_mean=-1, init_scale_std=0.1):
        """Returns a prior function with stateful parameters for EB models."""
        def prior_fn(dtype, shape, name, _, add_variable_fn):
            """A prior for the variational layers."""
            untransformed_scale = add_variable_fn(
                name=name + '_untransformed_scale',
                shape=shape,
                # shape=(1,)
                initializer=tf.random_normal_initializer(
                    mean=init_scale_mean, stddev=init_scale_std),
                dtype=dtype,
                trainable=True)
                # trainable=False
            loc = add_variable_fn(
                name=name + '_loc',
                initializer=keras.initializers.Zeros(),
                shape=(1,),
                # shape=shape
                dtype=dtype,
                trainable=False)
                # trainable=True
            scale = tfp.util.DeferredTensor(
                untransformed_scale,
                lambda x: (np.finfo(dtype.as_numpy_dtype).eps + tf.nn.softplus(x)))
            dist = tfd.Normal(loc=loc, scale=scale)
            batch_ndims = tf.size(input=dist.batch_shape_tensor())
            return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
        return prior_fn