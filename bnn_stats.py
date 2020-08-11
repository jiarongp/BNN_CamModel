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


fname = 'results/' + params.database + '/bnn_stats.log'
roc_fig = 'results/' + params.database + '/bnn_stats.png'
ckpt_dir = 'ckpts/' + params.database + '/' + params.model_type

# import BNN model
train_size = 0
for m in params.brand_models:
    train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
num_test_steps = int((train_size + params.BATCH_SIZE/2 - 1) // (params.BATCH_SIZE/2))

model = model_lib.BNN(train_size)
ckpt = tf.train.Checkpoint(
    step=tf.Variable(1), 
    optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
    net=model)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

fn = lambda x: tf.py_function(dp.parse_image, inp=[x], Tout=[tf.float32, tf.float32])
in_dataset = (tf.data.Dataset.list_files(params.patch_dir + '/test/*/*')
              .repeat()
              .map(dp.parse_image if params.database == 'dresden'
                   else fn, num_parallel_calls=AUTOTUNE 
                   if params.database == 'dresden' else None)
              .batch(32)
              .prefetch(buffer_size=AUTOTUNE))

jpeg_fn = lambda x: tf.py_function(dp.parse_image, inp=[x, 'jpeg'], Tout=[tf.float32, tf.float32])
jpeg_dataset = (tf.data.Dataset.list_files(params.patch_dir+ '/test/*/*')
                .repeat()
                .map(functools.partial(dp.parse_image, post_processing='jpeg')
                     if params.database == 'dresden' else jpeg_fn,
                     num_parallel_calls=AUTOTUNE 
                     if params.database == 'dresden' else None)
                .batch(32)
                .prefetch(buffer_size=AUTOTUNE))

# unseen_dataset = (tf.data.Dataset.list_files(params.unseen_images_dir+ '/*/*')
#                   .repeat()
#                   .map(dp.parse_image, 
#                        num_parallel_calls=AUTOTUNE 
#                        if params.database == 'dresden' else None)
#                   .batch(32)
#                   .prefetch(buffer_size=AUTOTUNE))

in_iter = iter(in_dataset)
jpeg_iter = iter(jpeg_dataset)
# unseen_iter = iter(unseen_dataset)

mc_s_prob_in = []
mc_s_prob_out = []
for i in trange(params.num_monte_carlo):
    softmax_prob_in, softmax_prob_out = [], []
    for step in range(40):
        in_images, _ = in_iter.get_next()
        out_images, _ = jpeg_iter.get_next()
        logits_in = model(in_images)
        logits_out = model(out_images)
        s_prob_in = tf.nn.softmax(logits_in)
        s_prob_out = tf.nn.softmax(logits_out)

        softmax_prob_in.extend(s_prob_in)
        softmax_prob_out.extend(s_prob_out)
    mc_s_prob_in.append(softmax_prob_in)
    mc_s_prob_out.append(softmax_prob_out)

mc_s_prob_in = np.asarray(mc_s_prob_in)
mc_s_prob_out = np.asarray(mc_s_prob_out)

in_log_prob, in_epistemic_all = utils.image_uncertainty(mc_s_prob_in)
out_log_prob, out_epistemic_all = utils.image_uncertainty(mc_s_prob_out)

log_prob_unseen = [in_log_prob, out_log_prob]
epistemic_unseen = [in_epistemic_all, out_epistemic_all]

targets = [('log_prob, jpeg models', log_prob_unseen), 
            ('epistemic, jpeg models', epistemic_unseen)]

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
canvas.print_figure(roc_fig, format='png')
print('saved {}'.format(roc_fig))

