import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import numpy as np
import params
import utils
import os
import tensorflow_probability as tfp
import data_preparation
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

tfd = tfp.distributions
keras = tf.keras
AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

NUM_CLASSES = len(params.brand_models)

class constrain_conv(tf.keras.models.Model, tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(constrain_conv, self).__init__()
        self.layer = model.layers[0]
        self.pre_weights = None

    def on_batch_begin(self, batch, logs={}):
        weights = self.layer.get_weights()[0]
        bias = self.layer.get_weights()[1]
        if self.pre_weights is None or np.all(self.pre_weights != weights):
            weights = weights * 10000
            weights[2, 2, :, :] = 0
            s = np.sum(weights, axis=(0,1))
            for i in range(3):
                weights[:, :, 0, i] /= s[0, i]
            weights[2, 2, :, :] = -1
            self.pre_weights = weights
        self.layer.set_weights([weights, bias])

if __name__ == '__main__':
    utils.set_logger('keras.log')

    logging.info("Creating the datasets...")
    data_preparation.collect_split_extract(parent_dir=params.patches_dir, download_images=False)

    train_size = 0
    val_size = 0
    num_images_per_class = []
    class_weight = {}
    for m in params.brand_models:
        num_images = len(os.listdir(os.path.join(params.patches_dir, 'train', m)))
        num_images_per_class.append(num_images)
        train_size += num_images
        val_size += len(os.listdir(os.path.join(params.patches_dir, 'val', m)))
        
    num_batches = (train_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE

    for n in range(len(params.brand_models)):
        class_weight[n] = (1 / num_images_per_class[n])*(train_size)/2.0
        logging.info('Weight for class {}: {:.2f}'.format(n, class_weight[n]))

    divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(100 * train_size, dtype=tf.float32))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(3, (5, 5), 
                            padding='same'),
        tfp.layers.Convolution2DFlipout(96, 
            kernel_size=7, strides=2, padding='SAME', 
            # kernel_prior_fn=eb_prior_fn,
            # kernel_divergence_fn=divergence_fn,
            activation=tf.nn.selu),
        tf.keras.layers.MaxPool2D(
            pool_size=[3, 3], strides=2,
            padding='SAME'),
        tfp.layers.Convolution2DFlipout(
            64, kernel_size=5, strides=1,
            padding='SAME', 
            # kernel_prior_fn=eb_prior_fn,
            kernel_divergence_fn=divergence_fn,
            activation=tf.nn.selu),
        tf.keras.layers.MaxPool2D(
            pool_size=[3, 3], strides=2),
        tfp.layers.Convolution2DFlipout(
            64, kernel_size=5, strides=1,
            padding='SAME', 
            # kernel_prior_fn=eb_prior_fn,
            kernel_divergence_fn=divergence_fn,
            activation=tf.nn.selu),
        tf.keras.layers.MaxPool2D(
            pool_size=[3, 3], strides=2),
        tfp.layers.Convolution2DFlipout(
            128, kernel_size=1, strides=1,
            padding='SAME', 
            # kernel_prior_fn=eb_prior_fn,
            kernel_divergence_fn=divergence_fn,
            activation=tf.nn.selu),
        tf.keras.layers.MaxPool2D(
            pool_size=[3, 3], strides=2,),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(
            200, kernel_divergence_fn=divergence_fn,
            activation=tf.nn.selu),
        tfp.layers.DenseFlipout(
            200, kernel_divergence_fn=divergence_fn,
            activation=tf.nn.selu),
        tfp.layers.DenseFlipout(
            NUM_CLASSES, 
            # kernel_prior_fn=eb_prior_fn,
            kernel_divergence_fn=divergence_fn)])

    train_ds = (tf.data.Dataset.list_files(params.patches_dir + '/train/*/*')
        .shuffle(buffer_size=1000)
        .map(data_preparation._parse_image, num_parallel_calls=AUTOTUNE)
        .batch(params.BATCH_SIZE)
    )
    val_ds = (tf.data.Dataset.list_files(params.patches_dir + '/val/*/*')
        .map(data_preparation._parse_image, num_parallel_calls=AUTOTUNE)
        .batch(params.BATCH_SIZE)
    )

    # @tf.function
    def kl_loss(y_true, y_pred):
        kl = tf.reduce_sum(model.losses)
        return kl

    # @tf.function
    def nll_loss(y_true, y_pred):
        nll = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        return nll

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy', kl_loss, nll_loss],
                experimental_run_tf_function=False)
    model.build(input_shape=[None, 256, 256, 1])
    constrain_conv_layer = constrain_conv(model)
    model.summary()

    # Create a callback that saves the model's weights
    ckpts_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./ckpts/0_kl/',
                                                        save_weights_only=True,
                                                        monitor='val_accuracy', mode='max',
                                                        save_best_only=True,
                                                        verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
                        # Stop training when `val_loss` is no longer improving
                        monitor="val_loss",
                        # "no longer improving" being defined as "no better than 1e-2 less"
                        min_delta=1e-2,
                        # "no longer improving" being further defined as "for at least 2 epochs"
                        patience=2,
                        verbose=1,)
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    logging.info('... Training convolutional neural network\n')
    history = model.fit(train_ds, epochs=10, 
                        callbacks=[constrain_conv_layer, ckpts_callback, tensorboard_callback, early_stopping], 
                        validation_data=val_ds, class_weight=class_weight)