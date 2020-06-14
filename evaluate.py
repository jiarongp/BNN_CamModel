import logging
logging.getLogger('tensorflow').disabled = True
import utils
import data_preparation
import os
import params
from tqdm import trange
import model_lib
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def evaluate():
    utils.set_logger('evaluate.log')

    test_size = 0
    for m in params.brand_models:
        test_size += len(os.listdir(os.path.join(params.patches_dir, 'test', m)))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_iterator = data_preparation.build_dataset('test')

    # Define the model
    logging.info("Creating the model...")
    model = model_lib.BNN(num_examples_per_epoch=test_size)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    # Restore the weights
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, './ckpts/tmp', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    logging.info("Restored from {}".format(manager.latest_checkpoint))

    @tf.function
    def test_step(images, label):
        with tf.GradientTape() as tape:
            logits = model(images)
            neg_log_likelihood = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True)
            kl = sum(model.losses)
            loss = neg_log_likelihood + kl
            
        test_loss.update_state(loss)  
        test_accuracy.update_state(labels, logits)  

    num_train_steps = (test_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    for step in trange(num_train_steps):
        images, labels = test_iterator.get_next()
        test_step(images, labels)

    logging.info('test loss: {:.3f}, test accuracy: {:.3%}\n'.format(test_loss.result(),
                                                            test_accuracy.result()))

if __name__ == '__main__':
    evaluate()