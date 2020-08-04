import tensorflow as tf
import numpy as np
import params
import functools
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

keras = tf.keras
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

    layer.set_weights([weights, bias])
    return pre_weights

class BNN(keras.Model):
    def __init__(self, kl_weight):
                #  eb_prior_fn, 
                #  examples_per_epoch):
        super(BNN, self).__init__()
        divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / 
                         tf.cast(kl_weight, dtype=tf.float32))

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
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.variational_conv2 = \
            tfp.layers.Convolution2DFlipout(
                                64, kernel_size=5,
                                strides=1, padding='same',
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.variational_conv3 = \
            tfp.layers.Convolution2DFlipout(
                                64, kernel_size=5,
                                strides=1, padding='same',
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.variational_conv4 = \
            tfp.layers.Convolution2DFlipout(
                                128, kernel_size=1,
                                strides=1, padding='same',
                                kernel_divergence_fn=divergence_fn,
                                activation='selu')
        self.dense1 = tfp.layers.DenseFlipout(200,
                                kernel_divergence_fn=divergence_fn)
        self.dense2 = tfp.layers.DenseFlipout(200,
                                kernel_divergence_fn=divergence_fn)
        self.dense3 = tfp.layers.DenseFlipout(NUM_CLASSES,
                                kernel_divergence_fn=divergence_fn)

    def call(self, x, training=False):
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

        if training:
            self.constrained_weights = constrain_conv(
                          self.constrained_conv, 
                          self.constrained_weights)
        return x

class vanilla(keras.Model):
    def __init__(self):
                #  eb_prior_fn, 
                #  examples_per_epoch):
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

    def call(self, x, training=False):
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

        if training:
            self.constrained_weights = constrain_conv(
                          self.constrained_conv, 
                          self.constrained_weights)
        return x
