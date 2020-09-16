import logging
logging.getLogger('tensorflow').disabled = True
import utils
import data_preparation as dp
import os
import params
import numpy as np
from tqdm import trange
import model_lib
import tensorflow as tf
keras = tf.keras
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def evaluate(log, result_dir, ckpt_dir):
    utils.set_logger(log)

    test_size = 0
    train_size = 0
    val_size = []
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
        test_size += len(os.listdir(os.path.join(params.patch_dir, 'test', m)))
        val_size.append(len(os.listdir(os.path.join(params.patch_dir, 'val', m))))
    num_test_steps = (test_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE

    @tf.function
    def test_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images)
            loss = keras.losses.categorical_crossentropy(labels,
                                                        logits, 
                                                        from_logits=True)
            softmax = tf.nn.softmax(logits)
        total = tf.math.reduce_sum(labels, axis=0).numpy()
        gt = tf.math.argmax(labels, axis=1)
        pred = tf.math.argmax(logits, axis=1)
        corr = labels[pred == gt]
        corr_count = tf.math.reduce_sum(corr, axis=0).numpy()

        test_loss.update_state(loss)  
        test_accuracy.update_state(labels, logits)
        return corr_count, total

    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')

    num_model = 10
    for n in trange(num_model):
        test_loss.reset_states()
        test_accuracy.reset_states()
        logging.info("Creating the dataset...")
        test_iterator = dp.build_dataset('test')
        logging.info("Creating the model...")
        model = model_lib.vanilla()
        corr_count = [0 for m in params.brand_models]

        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), 
            optimizer=keras.optimizers.Adam(lr=0.0001), 
            net=model)
        manager = tf.train.CheckpointManager(ckpt, 
                                            ckpt_dir + '_{}'.format(n), 
                                            max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        logging.info("Restored from {}".format(manager.latest_checkpoint))

        corr_ls, total_ls = [[0 for m in params.brand_models] for i in range(2)]
        for step in trange(num_test_steps):
            images, labels = test_iterator.get_next()
            c, t = test_step(images, labels)
            corr_ls += c
            total_ls += t
        
        logging.info('Ensemble CNN number {}, test loss: {:.3f}, test accuracy: {:.3%}'
                    .format(n,
                            test_loss.result(),
                            test_accuracy.result()))
        for m, c, t in zip(params.models, corr_ls, total_ls):
            logging.info('{} accuracy: {:.3%}'.format(m, c / t))
        logging.info('\n')


if __name__ == '__main__':
    log = os.path.join('results', params.database, 'ensemble')
    result_dir = os.path.join('results', params.database)
    ckpt_dir = os.path.join('ckpts', params.database, 'vanilla')
    for path in [log, result_dir, ckpt_dir]:
        p_dir = os.path.dirname(path)
        if not os.path.exists(p_dir):
            os.makedirs(p_dir)
    evaluate(log+'.log', result_dir, ckpt_dir)