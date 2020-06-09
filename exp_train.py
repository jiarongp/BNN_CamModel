import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
import tensorflow_probability as tfp
import os
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import params
import proc
import model
import utils
import cnn

tfd = tfp.distributions
keras = tf.keras

AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

_VALIDATION_STEPS = 100
examples_per_epoch = 0
for m in params.brand_models:
    examples_per_epoch += len(os.listdir(os.path.join(params.patches_dir, 'train', m)))


def build_model():
    """Builds a ResNet keras.models.Model."""
    # eb_prior_fn (callable): Empirical Bayes prior for use with TFP layers.
    eb_prior_fn = cnn.make_prior_fn_for_empirical_bayes(
                params.HParams['init_prior_scale_mean'], 
                params.HParams['init_prior_scale_std'])


    keras_in = keras.layers.Input(shape=(256, 256, 1))
    net = cnn.bnn(inputs=keras_in, 
                  std_prior_scale=params.HParams['std_prior_scale'], 
                  eb_prior_fn=eb_prior_fn, 
                  examples_per_epoch=examples_per_epoch)

    divergence_fn = cnn.make_divergence_fn_for_empirical_bayes(
        params.HParams['std_prior_scale'], examples_per_epoch)

    keras_out = tfp.layers.DenseReparameterization(
        params.NUM_CLASSES,
        kernel_prior_fn=eb_prior_fn,
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
            loc_initializer=keras.initializers.he_normal()),
        kernel_divergence_fn=divergence_fn)(net)

    return keras.models.Model(inputs=keras_in, outputs=keras_out)


def _make_lr_scheduler(init_lr):
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
        logging.info('Learning rate=%f for epoch=%d ', rate, epoch)

        return rate

    return keras.callbacks.LearningRateScheduler(schedule_fn)


def build_and_train():
    # download, split dataset and extract patches
    class_weight = utils.collect_split_extract()
    # group images and labels into dataset
    # train_ds, val_ds, test_ds = utils.build_dataset()
    train_set = tf.data.Dataset.list_files(params.patches_dir + '/train/*/*')
    train_ds = train_set.map(utils.parse_image, num_parallel_calls=AUTOTUNE)
    val_set = tf.data.Dataset.list_files(params.patches_dir + '/val/*/*')
    val_ds = train_set.map(utils.parse_image, num_parallel_calls=AUTOTUNE)

    init_learning_rate = 0.001189
    model = build_model()

    model.compile(
        keras.optimizers.Adam(init_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir='./log', write_graph=False)
    lr_scheduler = _make_lr_scheduler(init_learning_rate)
    bs = 32
    val_ds = val_ds.take(bs * _VALIDATION_STEPS).repeat().batch(bs)
    model.fit(
        train_ds.repeat().shuffle(10*bs).batch(bs),
        steps_per_epoch=examples_per_epoch // bs,
        epochs=50,
        validation_data=val_ds,
        validation_steps=_VALIDATION_STEPS,
        callbacks=[tensorboard_cb, lr_scheduler, cnn.constrain_layer(model)],
        class_weight=class_weight
    )

if __name__ == "__main__":
    build_and_train()