import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import os
import params
import data_preparation as dp
import model_lib
import utils
import datetime
from tqdm import trange
keras = tf.keras
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def build_and_train(log, tb_log, ckpt_dir):

    utils.set_logger(log)
    logging.info("Creating the datasets...")
    dp.collect_split_extract(parent_dir=params.patch_dir)

    train_size = 0
    val_size = []
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
        val_size.append(len(os.listdir(os.path.join(params.patch_dir, 'val', m))))
    num_train_steps = (train_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    num_val_steps = (sum(val_size) + params.BATCH_SIZE - 1) // params.BATCH_SIZE

    
    class_imbalance = False if params.even_database else True
    train_iterator = dp.build_dataset('train', class_imbalance=class_imbalance)
    val_iterator = dp.build_dataset('val')
    
    loss_object = keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    train_loss = keras.metrics.Mean(name='train_loss')
    train_acc = keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = keras.metrics.Mean(name='test_loss')
    val_acc = keras.metrics.CategoricalAccuracy(name='test_accuracy')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(tb_log + current_time, 'train')
    val_log_dir = os.path.join(tb_log + current_time, 'val')
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)

    models = []
    num_model = 10
    for n in trange(num_model):
        best_acc = 1.0 / len(params.brand_models)
        best_loss = 10000
        stop_count = 0
        train_log_dir = os.path.join(tb_log + '_{}'.format(n) + current_time, 'train')
        val_log_dir = os.path.join(tb_log + '_{}'.format(n) + current_time, 'val')
        train_writer = tf.summary.create_file_writer(train_log_dir)
        val_writer = tf.summary.create_file_writer(val_log_dir)

        model = model_lib.vanilla()
        model.build(input_shape=(None, 256, 256, 1))

        ckpt = tf.train.Checkpoint(
                step=tf.Variable(1),
                optimizer=optimizer,
                net=model)
        manager = tf.train.CheckpointManager(ckpt, 
                                            ckpt_dir + '_{}'.format(n), 
                                            max_to_keep=3)

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_object(labels, logits)
            gradients = tape.gradient(loss, model.trainable_weights)
            if step % 100 == 0:
                with train_writer.as_default():
                    for grad, t_w in zip(gradients, model.trainable_weights):
                        if 'kernel' in t_w.name:
                            tf.summary.histogram(t_w.name, grad, offset+step)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            train_loss.update_state(loss)
            train_acc.update_state(labels, logits)

        @tf.function
        def val_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_object(labels, logits)
            total = tf.math.reduce_sum(labels, axis=0).numpy()
            gt = tf.math.argmax(labels, axis=1)
            pred = tf.math.argmax(logits, axis=1)
            corr = labels[pred == gt]
            corr_count = tf.math.reduce_sum(corr, axis=0).numpy()
            val_loss.update_state(loss)
            val_acc.update_state(labels, logits)
            return corr_count, total

        logging.info('... Training number {} convolutional neural network\n'
                     .format(n))
        for epoch in range(params.NUM_EPOCHS):
            offset = epoch * num_train_steps
            val_loss.reset_states()
            val_acc.reset_states()
            train_loss.reset_states()
            train_acc.reset_states()

            for step in trange(num_train_steps):
                images, labels = train_iterator.get_next()
                train_step(images, labels)
                train_writer.flush()

                if epoch == 0 and step == 0:
                    with val_writer.as_default():
                        tf.summary.scalar('loss',  train_loss.result(), step=offset)
                        tf.summary.scalar('accuracy', train_acc.result(), step=offset)
                        val_writer.flush()

                with train_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=offset+step)
                    tf.summary.scalar('accuracy', train_acc.result(), step=offset+step)
                    train_writer.flush()

                if (step+1) % 100 == 0:
                    logging.info('Epoch: {}, Batch index: {}, '
                            'train loss: {:.3f}, train accuracy: {:.3%}'
                            .format(epoch, step + 1, 
                            train_loss.result(), 
                            train_acc.result()))

            corr_ls, total_ls = [[0 for m in params.brand_models] for i in range(2)]
            for step in trange(num_val_steps):
                images, labels = val_iterator.get_next()
                c, t = val_step(images, labels)
                corr_ls += c
                total_ls += t

            with val_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=offset+num_train_steps)
                tf.summary.scalar('accuracy', val_acc.result(), step=offset+num_train_steps)
                val_writer.flush()

            logging.info('val loss: {:.3f}, validation accuracy: {:.3%}'.format(
                    val_loss.result(), val_acc.result()))

            for m, c, t in zip(params.models, corr_ls, total_ls):
                logging.info('{} accuracy: {:.3%}'.format(m, c / t))
            logging.info('\n')

            ckpt.step.assign_add(1)
            if val_loss.result() <= best_loss:
                save_path = manager.save()
                best_acc = val_acc.result()
                best_loss = val_loss.result()
                stop_count = 0
                logging.info("Saved checkpoint for epoch {}: {}\n".format(epoch, save_path))
            else:
                stop_count += 1
            if stop_count >= 3:
                break

        logging.info('\nFinished number {} training\n'.format(n))

if __name__ == '__main__':
    log = os.path.join('results', params.database, 'vanilla')
    tb_log = os.path.join('logs', params.database, 'vanilla')
    ckpt_dir = os.path.join('ckpts', params.database, 'vanilla')
    for path in [log, tb_log, ckpt_dir]:
        p_dir = os.path.dirname(path)
        if not os.path.exists(p_dir):
            os.makedirs(p_dir)
    build_and_train(log+'.log', tb_log, ckpt_dir)