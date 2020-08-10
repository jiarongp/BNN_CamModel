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


def bnn_evaluate(log):
    utils.set_logger(log)
    num_monte_carlo = 30
    test_size = 0
    train_size = 0
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
        test_size += len(os.listdir(os.path.join(params.patch_dir, 'test', m)))
    num_test_steps = (test_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_iterator = dp.build_dataset('test')

    # Define the model
    logging.info("Creating the model...")
    model = model_lib.BNN(train_size)
    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')
    corr_count, total = [[0 for m in params.brand_models] for i in range(2)]

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, 
                                         './ckpts/BNN_num_examples_4', 
                                         max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    logging.info("Restored from {}".format(manager.latest_checkpoint))

    @tf.function
    def test_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images)
            nll = keras.losses.categorical_crossentropy(labels,
                                                        logits, 
                                                        from_logits=True)
            kl = sum(model.losses)
            loss = nll + kl
            # accuracy for each class
            for logit, label in zip(labels, logits):
                y_true = tf.math.argmax(label)
                y_pred = tf.math.argmax(logit)
                total[y_true] += 1
                if y_true == y_pred:
                    corr_count[y_true] += 1

        test_loss.update_state(loss)  
        test_accuracy.update_state(labels, logits)  

    for step in trange(num_test_steps):
        images, labels = test_iterator.get_next()
        test_step(images, labels)

        if step % 50 == 0:
            probs, heldout_log_prob = utils.compute_probs(model, images)
            logging.info(' ... Held-out nats: {:.3f}\n'.format(heldout_log_prob))
            # transform from onehot to strings
            index = tf.argmax(labels.numpy(), axis=1)
            labels = [params.brand_models[i] for i in index]
            utils.plot_heldout_prediction(images, labels, probs,
                                  fname='results/step{}_pred.png'.
                                        format(step),
                                  title='mean heldout logprob {:.2f}'
                                  .format(heldout_log_prob))                                                 

    logging.info('test loss: {:.3f}, test accuracy: {:.3%}\n'.format(test_loss.result(),
                                                        test_accuracy.result()))
    for m, c, t in zip(params.models, corr_count, total):
        logging.info('{} accuracy: {:.3%}'.format(m, c / t))    

    names = [layer.name for layer in model.layers
            if 'flipout' in layer.name]
    qm_vals = [layer.kernel_posterior.mean() 
            for layer in model.layers
            if 'flipout' in layer.name]
    qs_vals = [layer.kernel_posterior.stddev() 
            for layer in model.layers
            if 'flipout' in layer.name]

    utils.plot_weight_posteriors(names, qm_vals, qs_vals, 
                                 fname="results/trained_weight.png")
    logging.info("\nmean of mean is {}, mean variance is {}".
                format(tf.reduce_mean(qm_vals[0]),
                tf.reduce_mean(qs_vals[0])))                              

def vanilla_evaluate(log):
    utils.set_logger(log)

    test_size = 0
    for m in params.brand_models:
        test_size += len(os.listdir(os.path.join(params.patch_dir, 'test', m)))
    num_test_steps = (test_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    # Create the input data pipeline
    logging.info("Start evalution...")
    test_iterator = dp.build_dataset('test')

    # Define the model
    logging.info("Creating the model...")
    model = model_lib.vanilla()
    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')
    corr_count, total = [[0 for m in params.brand_models] for i in range(2)]

    @tf.function
    def test_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images)
            loss = keras.losses.categorical_crossentropy(labels,
                                                         logits, from_logits=True)
        # accuracy for each class
        for logit, label in zip(labels, logits):
            y_true = tf.math.argmax(label)
            y_pred = tf.math.argmax(logit)
            total[y_true] += 1
            if y_true == y_pred:
                corr_count[y_true] += 1
        test_loss.update_state(loss)  
        test_accuracy.update_state(labels, logits)

    for step in trange(num_test_steps):
        images, labels = test_iterator.get_next()
        test_step(images, labels)                                           

    logging.info('test loss: {:.3f}, test accuracy: {:.3%}\n'.format(test_loss.result(),
                                                        test_accuracy.result()))
    for m, c, t in zip(params.models, corr_count, total):
        logging.info('{} accuracy: {:.3%}'.format(m, c / t))

if __name__ == '__main__':
    if params.model_type == 'bnn':
        log = 'results/' + params.database + '/bnn_num_examples.log'
        bnn_evaluate(log)
    elif params.model_tpye == 'vanilla':
        log = 'results/' + params.database + '/vanilla.log'
        vanilla_evaluate(log)