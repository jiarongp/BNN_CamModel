import numpy as np
import tensorflow as tf
import params
import data_preparation as dp
import utils
import os
import model_lib
import seaborn as sns
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure
from matplotlib.backends import backend_agg
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def stats(ckpt_dir, stats_fig, fname, experiment, quality_ls):
    # collect data from unseen models
    dp.collect_unseen()

    # load model
    train_size = 0
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
    # import BNN model
    model = model_lib.bnn(train_size)

    ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), 
            optimizer=tf.keras.optimizers.RMSprop(lr=params.HParams['init_learning_rate']),
            net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    ds, num_test_batches = dp.aligned_ds(params.patch_dir, 
                                        params.brand_models)

    in_dataset = (tf.data.Dataset.from_tensor_slices(ds)
                    .repeat()
                    .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                    .batch(params.BATCH_SIZE)
                    .prefetch(buffer_size=AUTOTUNE))
    in_iter = iter(in_dataset)
    print("... In-distribution MC Statistics")
    in_entropy, in_epistemic = utils.mc_in_stats(in_iter, model, 
                                                num_test_batches, 
                                                params.num_monte_carlo)
    # Plotting ROC and PR curves
    fig = figure.Figure(figsize=(5, 10))
    canvas = backend_agg.FigureCanvasAgg(fig)
    fz = 15
    opt_list = []
    sns.set_style("darkgrid")
    inverse = False
    for quality in quality_ls:
        print("... {} Out-of-distribution MC Statistics".format(experiment))
        out_ds = dp.post_processing(ds, experiment, quality)
        out_dataset = (tf.data.Dataset.from_tensor_slices(out_ds)
                        .repeat()
                        .map(dp.parse_image,
                            num_parallel_calls=AUTOTUNE)
                        .batch(params.BATCH_SIZE)
                        .prefetch(buffer_size=AUTOTUNE))
        out_iter = iter(out_dataset)
        out_entropy, out_epistemic, out_class_count = \
                        utils.mc_out_stats(out_iter, model, 
                                            num_test_batches,
                                            params.num_monte_carlo)

        target = [('{} entropy '.format(experiment), 
                [in_entropy, out_entropy]),
                ('{} epistemic'.format(experiment), 
                [in_epistemic, out_epistemic])]
                    
        for i, (plotname, (safe, risky)) in enumerate(target):
            ax = fig.add_subplot(2, 1, i+1)
            fpr, tpr, opt, auroc = utils.roc_pr_curves(safe, risky, inverse)
            opt_list.append(opt)
            acc = np.sum((risky > opt[2]).astype(int)) / risky.shape[0]
            msg = (plotname + '\n'
                "false positive rate: {:.3%}, "
                "true positive rate: {:.3%}, "
                "threshold: {}, "
                "acc: {:.3%}\n".format(opt[0], opt[1], opt[2], acc))
            print(msg)

            ax.plot(fpr, tpr, '-',
                    label='{} quality {}: {}'.format(experiment, quality, auroc),
                    lw=2)
            ax.plot([0, 1], 'k-', lw=1)
            ax.legend(fontsize=10)
            ax.set_title(plotname, fontsize=fz)
            ax.set_xlabel("FPR", fontsize=fz)
            ax.set_ylabel("TPR", fontsize=fz)
            ax.grid(True)

    # fig.suptitle('ROC curve of uncertainty binary detector (correct / in-distribution as positive)', y=1.07, fontsize=25)
    fig.tight_layout()
    canvas.print_figure(stats_fig, format='png')
    print('saved {}'.format(stats_fig))

if __name__ == '__main__':
    experiment = 'noise'
    ckpt_dir = 'ckpts/' + params.database + '/' + params.model_type
    stats_fig = ('results/' + params.database + '/' + params.model_type +
                 '_'+ experiment + '_stats.png')
    fname = ('results/' + params.database + '/' + params.model_type + 
             '_degradation_stats.log')
    if experiment == 'jpeg':
        quality_ls = [90, 80, 70, 60, 50]
    elif experiment == 'blur':
        # fixed windows size and change window size
        quality_ls = [0.1, 0.5, 1.1, 1.5, 2.0]
    elif experiment == 'noise':
        quality_ls = [0.0, 0.1, 0.5, 1.5, 2.0]

    stats(ckpt_dir, stats_fig, fname, experiment, quality_ls)