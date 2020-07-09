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
    data_preparation.collect_split_extract(download_images=False, 
                                           parent_dir=params.patches_dir)

    train_size = 0
    val_size = 0
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patches_dir, 'train', m)))
        val_size += len(os.listdir(os.path.join(params.patches_dir, 'val', m)))
    num_train_steps = (train_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    num_val_steps = (val_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE

    # variables for early stopping and saving best models
    # random guessing
    best_acc = 1.0 / len(params.brand_models)
    best_loss = 10000
    stop_count = 0

    train_iterator = data_preparation.build_dataset('train', class_imbalance=True)
    val_iterator = data_preparation.build_dataset('val')

    model = model_lib.BNN(train_size)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate'])

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    kl_loss = tf.keras.metrics.Mean(name='kl_loss')
    nll_loss = tf.keras.metrics.Mean(name='nll_loss')
    val_loss = tf.keras.metrics.Mean(name='test_loss')
    val_acc = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)


    # save model to a checkpoint
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, './ckpts/BNN_2', max_to_keep=3)
    # ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            nll =loss_object(labels, logits)
            kl = sum(model.losses)
            loss = nll + kl
            kl_loss.update_state(kl)  
            nll_loss.update_state(nll)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        train_loss.update_state(loss)  
        train_acc.update_state(labels, logits)

    @tf.function
    def val_step(images, label):
        with tf.GradientTape() as tape:
            logits = model(images)
            neg_log_likelihood =loss_object(labels, logits)
            kl = sum(model.losses)
            loss = neg_log_likelihood + kl
        val_loss.update_state(loss)
        val_acc.update_state(labels, logits)

    logging.info('... Training convolutional neural network\n')
    for epoch in range(params.NUM_EPOCHS):
        offset = epoch * num_train_steps

        val_loss.reset_states()
        val_acc.reset_states()
        train_loss.reset_states()
        train_acc.reset_states()
        kl_loss.reset_states()
        nll_loss.reset_states()

        for step in trange(num_train_steps):
            images, labels = train_iterator.get_next()
            train_step(images, labels)

            if epoch == 0 and step == 0:
                model.summary()
                with val_writer.as_default():
                    tf.summary.scalar('loss',  train_loss.result(), step=offset)
                    tf.summary.scalar('accuracy', train_acc.result(), step=offset)
                    val_writer.flush()

            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=offset + step)
                tf.summary.scalar('accuracy', train_acc.result(), step=offset + step)
                tf.summary.scalar('kl_loss', kl_loss.result(), step=offset + step)
                tf.summary.scalar('nll_loss', nll_loss.result(), step=offset + step)
                train_writer.flush()

            if (step+1) % 150 == 0:
                logging.info('Epoch: {}, Batch index: {}, '
                            'train loss: {:.3f}, train accuracy: {:.3%}'.format(epoch, step + 1, 
                        train_loss.result(), train_acc.result()))

        for step in trange(num_val_steps):
            images, labels = val_iterator.get_next()
            val_step(images, labels)

        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=offset+num_train_steps)
            tf.summary.scalar('accuracy', val_acc.result(), step=offset+num_train_steps)
            val_writer.flush()

        logging.info('val loss: {:.3f},validation accuracy: {:.3%}\n'.format(
                val_loss.result(), val_acc.result()))

        # save the best model regarding to train acc
        ckpt.step.assign_add(1)

        if val_acc.result() >= best_acc and \
        val_loss.result() <= best_loss:
            save_path = manager.save()
            best_acc = val_acc.result()
            best_loss = val_loss.result()
            stop_count = 0
            logging.info("Saved checkpoint for epoch {}: {}\n".format(epoch, save_path))
        # early stopping
        else:
            stop_count += 1
        
        if stop_count >= params.patience:
            break

    logging.info('\nFinished training\n')

if __name__ == '__main__':
    build_and_train()