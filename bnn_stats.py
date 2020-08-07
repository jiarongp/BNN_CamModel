import model_lib
import os
import params
import data_preparation as dp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import utils
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure
from matplotlib.backends import backend_agg
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset = 'dresden'
patch_dir = (params.dresden_patches 
             if dataset == 'dresden' 
             else params.RAISE_patches)

# import BNN model
train_size = 0
for m in params.brand_models:
    train_size += len(os.listdir(os.path.join(patch_dir, 'train', m)))
num_test_steps = (train_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE 

model = model_lib.BNN(train_size)
ckpt = tf.train.Checkpoint(
    step=tf.Variable(1), 
    optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
    net=model)
manager = tf.train.CheckpointManager(ckpt, './ckpts/BNN_num_examples_3', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

in_dataset = (tf.data.Dataset.list_files(patch_dir + '/test/*/*')
              .repeat()
              .shuffle(buffer_size=1000)
              .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
              .batch(32)
              .prefetch(buffer_size=AUTOTUNE))
unseen_dataset = (tf.data.Dataset.list_files(params.unseen_images_dir+ '/*/*')
                  .repeat()
                  .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                  .batch(32)
                  .prefetch(buffer_size=AUTOTUNE))
in_iter = iter(in_dataset)
unseen_iter = iter(unseen_dataset)

fname = 'results/bnn_stats.log'

mc_s_prob_in = []
mc_s_prob_out = []
for i in trange(params.num_monte_carlo):
    softmax_prob_in, softmax_prob_out = [], []
    for step in range(50):
        in_images, _ = in_iter.get_next()
        out_images, _ = unseen_iter.get_next()
        logits_in = model(in_images)
        logits_out = model(out_images)
        s_prob_in = tf.nn.softmax(logits_in)
        s_prob_out = tf.nn.softmax(logits_out)

        softmax_prob_in.extend(s_prob_in)
        softmax_prob_out.extend(s_prob_out)
    mc_s_prob_in.append(softmax_prob_in)
    mc_s_prob_out.append(softmax_prob_out)

mc_s_prob_in = np.asarray(mc_s_prob_in)
mc_s_prob_out = np.asarray(mc_s_prob_in)

in_log_prob, in_epistemic_all = utils.image_uncertainty(mc_s_prob_in)
out_log_prob, out_epistemic_all = utils.image_uncertainty(mc_s_prob_out)

log_prob_unseen = [in_log_prob, out_log_prob]
epistemic_unseen = [in_log_prob, out_log_prob]

targets = [('log_prob, unseen models', log_prob_unseen), 
            ('epistemic, unseen models', epistemic_unseen)]

# Plotting ROC and PR curves 
fig = figure.Figure(figsize=(10, 5))
canvas = backend_agg.FigureCanvasAgg(fig)
fz = 15
for i, (plotname, (safe, risky)) in enumerate(targets):
    ax = fig.add_subplot(1, 2, i+1)
    fpr, tpr, precision, recall, aupr, auroc = utils.roc_pr_curves(safe, risky)
    ax.plot(fpr, tpr, '-',
            label='Uncertainty, AUROC:{}'.format(auroc),
            lw=4)
    ax.plot([0, 1], 'k-', lw=3, label='Base rate(0.5)')
    ax.legend(fontsize=fz)
    ax.set_title(plotname, fontsize=fz)
    ax.set_xlabel("FPR", fontsize=fz)
    ax.set_ylabel("TPR", fontsize=fz)
    ax.grid(True)

fig.suptitle('ROC curve of uncertainty binary detector (correct / in-distribution as positive)', y=1.07, fontsize=30)
fig.tight_layout()
canvas.print_figure('results/bnn_stats.png', format='png')
print('saved {}'.format('results/bnn_stats.png'))

