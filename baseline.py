import numpy as np
import tensorflow as tf
import params
import data_preparation as dp
import utils
import pandas as pd
import os
import model_lib
import functools
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure
from matplotlib.backends import backend_agg
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
AUTOTUNE = tf.data.experimental.AUTOTUNE

BNN = False

# switch between BNN and vanilla CNN
if BNN:
    model = model_lib.BNN(train_size)

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, './ckpts/BNN_num_examples_2', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
else:
    # Load and Compile the model
    model = model_lib.vanilla()

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, './ckpts/vanilla', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

# build test iterator
test_size = 0
for m in params.brand_models:
    test_size += len(os.listdir(os.path.join(params.patches_dir, 'test', m)))
num_test_steps = (test_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
test_iterator = dp.build_dataset('test')

# Right Wrong Distinction
softmax_prob_right, softmax_prob_wrong = utils.right_wrong_distinction(
                                         test_iterator, model, num_test_steps)

# build in & out of distribution dataset iterator
batch_size = np.int64(params.BATCH_SIZE / 2)
in_dataset = (tf.data.Dataset.list_files(params.patches_dir + '/test/*/*')
              .repeat()
              .shuffle(buffer_size=1000)
              .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
              .batch(batch_size)
              .prefetch(buffer_size=AUTOTUNE))
unseen_dataset = (tf.data.Dataset.list_files(params.unseen_images_dir+ '/*/*')
                  .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                  .batch(batch_size)
                  .prefetch(buffer_size=AUTOTUNE))
jpeg_dataset = (tf.data.Dataset.list_files(params.unseen_images_dir+ '/*/*')
                .map(functools.partial(dp.parse_image, post_processing='jpeg'), 
                     num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE))
blur_dataset = (tf.data.Dataset.list_files(params.unseen_images_dir+ '/*/*')
                .map(functools.partial(dp.parse_image, post_processing='blur'), 
                     num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE))
noise_dataset = (tf.data.Dataset.list_files(params.unseen_images_dir+ '/*/*')
                .map(functools.partial(dp.parse_image, post_processing='noise'), 
                     num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE))
in_iter = iter(in_dataset)
unseen_iter = iter(unseen_dataset)
jpeg_iter = iter(jpeg_dataset)
blur_iter = iter(blur_dataset)
noise_iter = iter(noise_dataset)

# In Out Distinction
# images from unseen models
unseen_prob_in, unseen_prob_out = utils.in_out_distinction(in_iter, unseen_iter, 
                                                           model, num_test_steps,
                                                           ood_name='UNSEEN MODEL')
# jpeg images
jpeg_prob_in, jpeg_prob_out = utils.in_out_distinction(in_iter, jpeg_iter, 
                                                       model, num_test_steps,
                                                       ood_name='JPEG')
# blurred images
blur_prob_in, blur_prob_out = utils.in_out_distinction(in_iter, blur_iter, 
                                                       model, num_test_steps,
                                                       ood_name='BLUR')
# blurred images
noise_prob_in, noise_prob_out = utils.in_out_distinction(in_iter, noise_iter, 
                                                         model, num_test_steps,
                                                         ood_name='NOISE')

# Bind softmax right/wrong distinction
s_p_rw = [np.asarray(softmax_prob_right), np.asarray(softmax_prob_wrong)]
s_p_io_unseen = [np.asarray(unseen_prob_in), np.asarray(unseen_prob_out)]
s_p_io_jpeg = [np.asarray(jpeg_prob_in), np.asarray(jpeg_prob_out)]
s_p_io_blur = [np.asarray(blur_prob_in), np.asarray(blur_prob_out)]
s_p_io_noise = [np.asarray(noise_prob_in), np.asarray(noise_prob_out)]

targets = [('right/wrong', s_p_rw),
           ('in/out, unseen models', s_p_io_unseen),
           ('in/out, jpeg', s_p_io_jpeg),
           ('in/out, blur', s_p_io_blur),
           ('in/out, noise', s_p_io_noise)]

# Plotting ROC and PR curves 
fig = figure.Figure(figsize=(25, 5))
canvas = backend_agg.FigureCanvasAgg(fig)
fz = 15
for i, (plotname, (safe, risky)) in enumerate(targets):
    ax = fig.add_subplot(1, 5, i+1)
    fpr, tpr, precision, recall, aupr, auroc = utils.roc_pr_curves(safe, risky)
    ax.plot(fpr, tpr, '-',
            label='Softmax, AUROC:{}'.format(auroc),
            lw=4)
    ax.plot([0, 1], 'k-', lw=3, label='Base rate(0.5)')
    ax.legend(fontsize=fz)
    ax.set_title(plotname, fontsize=fz)
    ax.set_xlabel("FPR", fontsize=fz)
    ax.set_ylabel("TPR", fontsize=fz)
    ax.grid(True)

    # ax = fig.add_subplot(2, 2, 2*i+2)
    # ax.plot(precision, recall, '-',
    #          label='Softmax, AUPR:{}'.format(aupr),
    #          lw=4)
    # no_skill = len(safe) / (len(safe) + len(risky))
    # ax.plot([0, 1], [no_skill, no_skill],'k-', lw=3, label='Base rate {}'.format(no_skill))
    # ax.legend(fontsize=fz)
    # ax.set_title(plotname, fontsize=fz)
    # ax.set_ylabel("Precision", fontsize=fz)
    # ax.set_xlabel("Recall", fontsize=fz)
    # ax.grid(True)

fig.suptitle('ROC curve of Softmax binary detector (correct / in-distribution as positive)', y=1.07, fontsize=30)
fig.tight_layout()
canvas.print_figure('results/baseline.png', format='png')
print('saved {}'.format('results/baseline.png'))



