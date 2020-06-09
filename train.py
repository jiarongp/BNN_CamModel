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
from tensorflow_probability import distributions as tfd

AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def build_and_train():
    # collect data if not downloaded
    data = pd.read_csv(params.dresden)
    data = data[([m in params.models for m in data['model']])]
    image_paths = utils.collect_dataset(data, params.dresden_images_dir)

    # split dataset in train, val and test
    split_ds, weights = utils.split_dataset(image_paths)
    class_weight = {}
    for i in range(len(params.brand_models)):
        class_weight[i] = weights[i]
    # extract patches from full-sized images
    for i in range(len(params.brand_models)):
        print("... Extracting patches from {} images".format(params.brand_models[i]))
        proc.patch(path=split_ds[i][0], dataset='train')
        proc.patch(path=split_ds[i][1], dataset='val')
        proc.patch(path=split_ds[i][2], dataset='test')
        print("... Done\n")
    # group images and labels into dataset
    train_ds, val_ds, test_ds = utils.build_dataset()

    NUM_EXAMPLES = 0
    for m in params.brand_models:
        NUM_EXAMPLES += len(os.listdir(os.path.join(params.patches_dir, 'train', m)))

    # eb_prior_fn (callable): Empirical Bayes prior for use with TFP layers.
    eb_prior_fn = model.make_prior_fn_for_empirical_bayes(params.HParams['init_prior_scale_mean'], 
                                                          params.HParams['init_prior_scale_std'])
    # bnn = model.create_model(NUM_EXAMPLES)
    bnn = model.BNN(params.HParams['std_prior_scale'],
                    eb_prior_fn, 
                    NUM_EXAMPLES)
    # Model compilation
    # The Keras API will then automatically add the Kullback-Leibler divergence (contained 
    # on the individual layers of the model), to the cross entropy loss, effectively 
    # calculating the (negated) Evidence Lower Bound Loss (ELBO)
    bnn.compile(
        tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])
    # bnn.build(input_shape=[None, params.IMG_WIDTH, params.IMG_HEIGHT, 1])

    # save model to a checkpoint
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=bnn)
    manager = tf.train.CheckpointManager(ckpt, './ckpts/bnn', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    with open("result/bnn_training.txt",'w', encoding = 'utf-8') as f:
        f.write('... Training convolutional neural network\n')
        print('... Training convolutional neural network')
        pre_weights = None

        for epoch in range(params.NUM_EPOCHS):
            epoch_accuracy, epoch_loss = [], []
            for step, (batch_x, batch_y) in enumerate(train_ds):
                step += 1
                batch_loss, batch_accracy = bnn.train_on_batch(batch_x, batch_y, class_weight=class_weight)
                epoch_accuracy.append(batch_accracy)
                epoch_loss.append(batch_loss)
                # constrained layer
                pre_weights = model.constrain_conv(bnn.layers[0], pre_weights)

                if step % 150 == 0:
                    f.write('Epoch: {}, Batch index: {}, '
                        'train loss: {:.3f}, train accuracy: {:.3%}\n'.format(epoch, step, 
                        tf.reduce_mean(epoch_loss), 
                        tf.reduce_mean(epoch_accuracy)))
                    print('Epoch: {}, Batch index: {}, '
                        'train loss: {:.3f}, train accuracy: {:.3%}'.format(epoch, step, 
                        tf.reduce_mean(epoch_loss), 
                        tf.reduce_mean(epoch_accuracy)))

            val_accuracy, val_loss = [], []
            for step, (batch_x, batch_y) in enumerate(val_ds):
                step += 1
                batch_loss, batch_accracy = bnn.test_on_batch(batch_x, batch_y)
                val_accuracy.append(batch_accracy)
                val_loss.append(batch_loss)
            f.write('Epoch: {}, validation loss: {:.3f}, validation accuracy: {:.3%}\n\n'.format(
                epoch, tf.reduce_mean(val_loss), tf.reduce_mean(val_accuracy)))
            print('Epoch: {}, validation loss: {:.3f}, validation accuracy: {:.3%}\n\n'.format(
                epoch, tf.reduce_mean(val_loss), tf.reduce_mean(val_accuracy)))

            # every 5 epoch save a check point
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 5 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}\n\n".format(int(ckpt.step), save_path))

        f.write('\nFinished training\n')
        print('\nFinished training\n')
        
        test_accuracy, test_loss = [], []
        for step, (batch_x, batch_y) in enumerate(test_ds):
            step += 1
            batch_loss, batch_accracy = bnn.test_on_batch(batch_x, batch_y)
            test_accuracy.append(batch_accracy)
            test_loss.append(batch_loss)
            
        f.write('test loss: {:.3f}, test accuracy: {:.3%}'.format(
        tf.reduce_mean(test_loss), 
        tf.reduce_mean(test_accuracy)))
        print('test loss: {:.3f}, test accuracy: {:.3%}'.format(
        tf.reduce_mean(test_loss), 
        tf.reduce_mean(test_accuracy)))

if __name__ == "__main__":
    build_and_train()