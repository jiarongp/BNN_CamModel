import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import os
import params
import data_preparation
import model_lib
import utils
import datetime
from tqdm import trange
keras = tf.keras
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def build_and_train():
    # Set the logger
    utils.set_logger('train.log')

    logging.info("Creating the datasets...")
    data_preparation.collect_split_extract(download_images=False)

    train_size = 0
    val_size = 0
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patches_dir, 'train', m)))
        val_size += len(os.listdir(os.path.join(params.patches_dir, 'val', m)))
    num_train_steps = (train_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    num_val_steps = (val_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE

    train_iterator = data_preparation.build_dataset('train', class_imbalance=True)
    val_iterator = data_preparation.build_dataset('val')

    model = model_lib.BNN(num_train_steps)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate'])

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    kl_loss = tf.keras.metrics.Mean(name='kl')
    neg_loss = tf.keras.metrics.Mean(name='neg_log_likelihood')
    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    # hist_log_dir = 'logs/histogram/' + current_time
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)
    # hist_writer = tf.summary.create_file_writer(hist_log_dir)


    # save model to a checkpoint
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, './ckpts/num_batches_eb', max_to_keep=3)
    # ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            neg_log_likelihood =loss_object(labels, logits)
            kl = sum(model.losses)
            loss = neg_log_likelihood + kl
            kl_loss.update_state(kl)  
            neg_loss.update_state(neg_log_likelihood)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_loss.update_state(loss)  
        train_accuracy.update_state(labels, logits)

    @tf.function
    def val_step(images, label):
        with tf.GradientTape() as tape:
            logits = model(images)
            neg_log_likelihood =loss_object(labels, logits)
            kl = sum(model.losses)
            loss = neg_log_likelihood + kl
        val_loss.update_state(loss)
        val_accuracy.update_state(labels, logits)

    logging.info('... Training convolutional neural network\n')
    for epoch in range(params.NUM_EPOCHS):
        offset = epoch * num_train_steps

        val_loss.reset_states()
        val_accuracy.reset_states()
        train_loss.reset_states()
        train_accuracy.reset_states()
        kl_loss.reset_states()
        neg_loss.reset_states()

        for step in trange(num_val_steps):
            images, labels = val_iterator.get_next()
            val_step(images, labels)

        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=offset)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=offset)
            val_writer.flush()

        logging.info('val loss: {:.3f},validation accuracy: {:.3%}\n'.format(
                val_loss.result(), val_accuracy.result()))

        for step in trange(num_train_steps):
            images, labels = train_iterator.get_next()
            train_step(images, labels)

            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=offset + step)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=offset + step)
                tf.summary.scalar('kl_loss', kl_loss.result(), step=offset + step)
                tf.summary.scalar('neg_loss', neg_loss.result(), step=offset + step)
                train_writer.flush()

            # with hist_writer.as_default():
            #     for i in model.trainable_weights:
            #         name = i.name.split(":")[0]
            #         value = i.value()
            #         tf.summary.histogram(name, value, step=offset + step)
            #     hist_writer.flush()

            if (step+1) % 150 == 0:
                logging.info('Epoch: {}, Batch index: {}, '
                            'train loss: {:.3f}, train accuracy: {:.3%}'.format(epoch, step + 1, 
                        train_loss.result(), train_accuracy.result()))

        # every 5 epoch save a check point
        ckpt.step.assign_add(1)
        val_accuracy.result()
        if int(ckpt.step) % 5 == 0:
            save_path = manager.save()
            logging.info("Saved checkpoint for step {}: {}\n".format(int(ckpt.step), save_path))

    logging.info('\nFinished training\n')

if __name__ == '__main__':
    build_and_train()