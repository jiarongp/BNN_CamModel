import numpy as np
import tensorflow as tf
import params
import data_preparation as dp
import utils
import os
import model_lib
import functools
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure
from matplotlib.backends import backend_agg
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
AUTOTUNE = tf.data.experimental.AUTOTUNE


# set a fix subset of total test dataset, so that:
# 1. each class has same size of test images
# 2. each monte carlo draw has the same images as input
# 3. random seed controls the how to sample this subset
def aligned_ds(test_dir, brand_models, num_batches=None, seed=42):
    # default for 'test' data
    np.random.seed(seed)
    image_paths, num_images, ds = [], [], []
    for model in brand_models:
        images = os.listdir(os.path.join(test_dir, 'test', model))
        paths = [os.path.join(test_dir, 'test', model, im) for im in images]
        image_paths.append(paths)
        num_images.append(len(images))
    # # of batches for one class
    class_batches = min(num_images) // params.BATCH_SIZE
    num_test_batches = len(brand_models) * class_batches
    # sometimes database has more data in 'test', some has more in 'unseen'
    if num_batches is not None:
        num_test_batches = min(num_test_batches, num_batches)

    for model, images in zip(params.brand_models, image_paths):
        np.random.shuffle(images)
        ds.extend(images[0:class_batches * params.BATCH_SIZE])

    return ds, num_test_batches


def stats(ckpt_dir, stats_fig, fname):
    # collect data from unseen models
    dp.collect_unseen()

    if params.model_type == 'bnn':
        train_size = 0
        for m in params.brand_models:
            train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
        # import BNN model
        model = model_lib.BNN(train_size)
    else:
        model = model_lib.vanilla()

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    ds, num_test_batches = aligned_ds(params.patch_dir, 
                                      params.brand_models)
    unseen_ds, num_unseen_batches = aligned_ds(params.unseen_dir, 
                                               params.unseen_brand_models,
                                               num_batches=num_test_batches)

    if params.model_type == 'vanilla':
        # Right Wrong Distinction
        test_iter = dp.build_dataset('test', 'dresden')
        softmax_prob_right, softmax_prob_wrong = utils.right_wrong_distinction(
                                                            test_iter, model, 
                                                            num_test_batches,
                                                            fname)

    in_dataset = (tf.data.Dataset.from_tensor_slices(ds)
                    .repeat()
                    .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                    .batch(params.BATCH_SIZE )
                    .prefetch(buffer_size=AUTOTUNE))
    
    unseen_in_dataset = (tf.data.Dataset.from_tensor_slices(unseen_ds)
                            .repeat()
                            .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                            .batch(params.BATCH_SIZE )
                            .prefetch(buffer_size=AUTOTUNE))

    unseen_out_dataset = (tf.data.Dataset.list_files(params.unseen_dir + '/test/*/*')
                            .repeat()
                            .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                            .batch(params.BATCH_SIZE )
                            .prefetch(buffer_size=AUTOTUNE))

    jpeg_dataset = (tf.data.Dataset.from_tensor_slices(ds)
                    .map(functools.partial(dp.parse_image, post_processing='jpeg'),
                         num_parallel_calls=AUTOTUNE)
                    .cache()
                    .repeat()
                    .batch(params.BATCH_SIZE )
                    .prefetch(buffer_size=AUTOTUNE))

    blur_dataset = (tf.data.Dataset.from_tensor_slices(ds)
                    .map(functools.partial(dp.parse_image, post_processing='blur'),
                         num_parallel_calls=AUTOTUNE)
                    .cache()
                    .repeat()
                    .batch(params.BATCH_SIZE)
                    .prefetch(buffer_size=AUTOTUNE))
    
    noise_dataset = (tf.data.Dataset.from_tensor_slices(ds)
                        .map(functools.partial(dp.parse_image, post_processing='jpeg'),
                             num_parallel_calls=AUTOTUNE)
                        .cache()
                        .repeat()
                        .batch(params.BATCH_SIZE)
                        .prefetch(buffer_size=AUTOTUNE))

    in_iter = iter(in_dataset)
    unseen_in_iter = iter(unseen_in_dataset)
    unseen_out_iter = iter(unseen_out_dataset)
    jpeg_iter = iter(jpeg_dataset)
    blur_iter = iter(blur_dataset)
    noise_iter = iter(noise_dataset)

    if params.model_type == 'vanilla':
        iod = utils.in_out_distinction
        unseen_prob_in, unseen_prob_out = iod(unseen_in_iter, 
                                              unseen_out_iter, 
                                              model, num_unseen_batches,
                                              'UNSEEN MODEL', fname)
        # jpeg images
        jpeg_prob_in, jpeg_prob_out = iod(in_iter, jpeg_iter,
                                          model, num_test_batches,
                                          'JPEG', fname)
        # blurred images
        blur_prob_in, blur_prob_out = iod(in_iter, blur_iter, 
                                          model, num_test_batches,
                                          'BLUR', fname)
        # blurred images
        noise_prob_in, noise_prob_out = iod(in_iter, noise_iter, 
                                            model, num_test_batches,
                                            'NOISE', fname)

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

    elif params.model_type == 'bnn':
        mc_iod = utils.mc_in_out_distinction
        print('number of unseen batches {}'.format(num_unseen_batches))
        print('number of test batches {}'.format(num_test_batches))

        unseen_log_prob, unseen_epistemic = mc_iod(unseen_in_iter,
                                                    unseen_out_iter,
                                                    model, num_unseen_batches,
                                                    params.num_monte_carlo,
                                                    'UNSEEN', fname)

        jpeg_log_prob, jpeg_epistemic = mc_iod(in_iter, jpeg_iter, 
                                                model, num_test_batches,
                                                params.num_monte_carlo,
                                                'JPEG', fname)

        blur_log_prob, blur_epistemic = mc_iod(in_iter, blur_iter, 
                                                model, num_test_batches,
                                                params.num_monte_carlo,
                                                'BLUR', fname)

        noise_log_prob, noise_epistemic = mc_iod(in_iter, noise_iter,
                                                model, num_test_batches,
                                                params.num_monte_carlo,
                                                'NOISE', fname)

        targets = [('log_prob, unseen models', unseen_log_prob), 
                    ('log_prob, jpeg models', jpeg_log_prob), 
                    ('log_prob, blur models', blur_log_prob), 
                    ('log_prob, noise models', noise_log_prob), 
                    ('epistemic, unseen models', unseen_epistemic),
                    ('epistemic, jpeg models', jpeg_epistemic),
                    ('epistemic, blur models', blur_epistemic),
                    ('epistemic, noise models', noise_epistemic)]

    # Plotting ROC and PR curves 
    fig = figure.Figure(figsize=(20, 10) 
                        if params.model_type == 'bnn' 
                        else (25, 5))
    canvas = backend_agg.FigureCanvasAgg(fig)
    fz = 15
    for i, (plotname, (safe, risky)) in enumerate(targets):
        if params.model_type == 'bnn':
            ax = fig.add_subplot(2, 4, i+1)
        else:
            ax = fig.add_subplot(1, 5, i+1)
        fpr, tpr, thresholdls, auroc = utils.roc_pr_curves(safe, risky)
        ax.plot(fpr, tpr, '-',
                label='AUROC:{}'.format(auroc),
                lw=4)
        ax.plot([0, 1], 'k-', lw=3, label='Base rate(0.5)')
        ax.legend(fontsize=fz)
        ax.set_title(plotname, fontsize=fz)
        ax.set_xlabel("FPR", fontsize=fz)
        ax.set_ylabel("TPR", fontsize=fz)
        ax.grid(True)
    fig.suptitle('ROC curve of uncertainty binary detector (correct / in-distribution as positive)', y=1.07, fontsize=25)
    fig.tight_layout()
    canvas.print_figure(stats_fig, format='png')
    print('saved {}'.format(stats_fig))


if __name__ == "__main__":
    ckpt_dir = 'ckpts/' + params.database + '/' + params.model_type
    stats_fig = 'results/' + params.database + '/' + params.model_type + '_stats.png'
    fname = 'results/' + params.database + '/' + params.model_type + '_stats.log'
    stats(ckpt_dir, stats_fig, fname)

