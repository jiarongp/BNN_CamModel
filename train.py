import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
import tensorflow_probability as tfp
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import params
import proc
import model
import utils
from tqdm import trange
tfd = tfp.distributions
AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def build_and_train():
    # Set the logger
    utils.set_logger('train.log')

    logging.info("Creating the datasets...")
    class_weights = utils.collect_split_extract()

    train_iterator = utils.build_dataset('train')
    val_iterator = utils.build_dataset('val')

    train_size = 0
    val_size = 0
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patches_dir, 'train', m)))
        val_size += len(os.listdir(os.path.join(params.patches_dir, 'val', m)))

    bnn = model.BNN(num_examples_per_epoch=train_size)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate'])

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    # val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    # save model to a checkpoint
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=bnn)
    manager = tf.train.CheckpointManager(ckpt, './ckpts/bnn', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = bnn(images, training=True)
            neg_log_likelihood =loss_object(labels, logits)
            kl = sum(bnn.losses)
            loss = neg_log_likelihood + kl
        gradients = tape.gradient(loss, bnn.trainable_weights)
        optimizer.apply_gradients(zip(gradients, bnn.trainable_weights))
        train_loss.update_state(loss)  
        train_accuracy.update_state(labels, logits)

    @tf.function
    def val_step(images, label):
        logits = bnn(images)
        val_accuracy.update_state(labels, logits)     

    logging.info('... Training convolutional neural network\n')

    for epoch in range(params.NUM_EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        # val_loss.reset_states()
        val_accuracy.reset_states()
        # Compute number of batches in one epoch (one full pass over the training set)
        num_train_steps = (train_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
        for step in trange(num_train_steps):
            images, labels = train_iterator.get_next()
            train_step(images, labels)

            if (step+1) % 150 == 0:
                logging.info('Epoch: {}, Batch index: {}, '
                            'train loss: {:.3f}, train accuracy: {:.3%}'.format(epoch, step + 1, 
                            train_loss.result(), train_accuracy.result()))

        num_val_steps = (val_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
        for step in trange(num_val_steps):
            images, labels = val_iterator.get_next()
            val_step(images, labels)
            
        logging.info('Epoch: {}, validation accuracy: {:.3%}\n\n'.format(
            epoch, val_accuracy.result()))

        # every 5 epoch save a check point
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 5 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}\n\n".format(int(ckpt.step), save_path))

    logging.info('\nFinished training\n')

if __name__ == '__main__':
    build_and_train()