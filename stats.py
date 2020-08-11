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


def bnn_stats(ckpt_dir, stats_fig, fname, download):
    # collect data from unseen models
    dp.collect_unseen(download=download)
    # import BNN model
    train_size = 0
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
    batch_size = np.int64(params.BATCH_SIZE / 2)
    num_test_steps = (train_size + batch_size - 1) // batch_size

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
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE))

    unseen_dataset = (tf.data.Dataset.list_files(params.unseen_dir + '/test/*/*')
                    .repeat()
                    .map(dp.parse_image if params.database == 'dresden' else fn, 
                        num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)  
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))

    jpeg_fn = lambda x: tf.py_function(dp.parse_image, inp=[x, 'jpeg'], Tout=[tf.float32, tf.float32])
    jpeg_dataset = (tf.data.Dataset.list_files(params.patch_dir+ '/test/*/*')
                    .repeat()
                    .map(functools.partial(dp.parse_image, post_processing='jpeg')
                        if params.database == 'dresden' else jpeg_fn,
                        num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))

    blur_fn = lambda x: tf.py_function(dp.parse_image, inp=[x, 'blur'], Tout=[tf.float32, tf.float32])
    blur_dataset = (tf.data.Dataset.list_files(params.patch_dir+ '/test/*/*')
                    .repeat()
                    .map(functools.partial(dp.parse_image, post_processing='blur') 
                        if params.database == 'dresden' else blur_fn,
                        num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))
    
    noise_fn = lambda x: tf.py_function(dp.parse_image, inp=[x, 'noise'], Tout=[tf.float32, tf.float32])
    noise_dataset = (tf.data.Dataset.list_files(params.patch_dir+ '/test/*/*')
                    .repeat()
                    .map(functools.partial(dp.parse_image, post_processing='noise')
                        if params.database == 'dresden' else noise_fn,
                        num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))

    in_iter = iter(in_dataset)
    unseen_iter = iter(unseen_dataset)
    jpeg_iter = iter(jpeg_dataset)
    blur_iter = iter(blur_dataset)
    noise_iter = iter(noise_dataset)


    unseen_log_prob, unseen_epistemic = utils.mc_in_out_distinction(
                                            in_iter, unseen_iter, 
                                            num_test_steps, # todo
                                            params.num_monte_carlo,
                                            'UNSEEN', fname)
    jpeg_log_prob, jpeg_epistemic = utils.mc_in_out_distinction(
                                            in_iter, jpeg_iter, 
                                            num_test_steps, 
                                            params.num_monte_carlo,
                                            'JPEG', fname)
    blur_log_prob, blur_epistemic = utils.mc_in_out_distinction(
                                            in_iter, blur_iter, 
                                            num_test_steps, 
                                            params.num_monte_carlo,
                                            'BLUR', fname)
    noise_log_prob, noise_epistemic = utils.mc_in_out_distinction(
                                            in_iter, noise_iter, 
                                            num_test_steps, 
                                            params.num_monte_carlo,
                                            'NOISE', fname)
                                            

    targets = [('log_prob, unseen models', unseen_log_prob), 
                ('epistemic, unseen models', unseen_epistemic),
                ('log_prob, jpeg models', jpeg_log_prob), 
                ('epistemic, jpeg models', jpeg_epistemic),
                ('log_prob, blur models', blur_log_prob), 
                ('epistemic, blur models', blur_epistemic),
                ('log_prob, noise models', noise_log_prob), 
                ('epistemic, noise models', noise_epistemic)]

    # Plotting ROC and PR curves 
    fig = figure.Figure(figsize=(20, 10))
    canvas = backend_agg.FigureCanvasAgg(fig)
    fz = 15
    for i, (plotname, (safe, risky)) in enumerate(targets):
        ax = fig.add_subplot(2, 4, i+1)
        fpr, tpr, precision, recall, aupr, auroc = utils.roc_pr_curves(safe, risky)
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


def vanilla_stats(ckpt_dir, stats_fig, fname, download):
    # collect data from unseen models
    dp.collect_unseen(download=download)
    # Load and Compile the model
    model = model_lib.vanilla()

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    # build test iterator
    test_size = 0
    for m in params.brand_models:
        test_size += len(os.listdir(os.path.join(params.patch_dir, 'test', m)))
    num_test_steps = (test_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    test_iterator = dp.build_dataset('test', 'dresden')

    # Right Wrong Distinction
    softmax_prob_right, softmax_prob_wrong = utils.right_wrong_distinction(
                                            test_iterator, model, num_test_steps,
                                            fname)


    # build in & out of distribution dataset iterator
    batch_size = np.int64(params.BATCH_SIZE / 2)
    fn = lambda x: tf.py_function(dp.parse_image, inp=[x], Tout=[tf.float32, tf.float32])
    in_dataset = (tf.data.Dataset.list_files(params.patch_dir + '/test/*/*')
                    .repeat()
                    .map(dp.parse_image if params.database == 'dresden' else fn, 
                         num_parallel_calls=AUTOTUNE 
                         if params.database == 'dresden' else None) 
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))

    unseen_dataset = (tf.data.Dataset.list_files(params.unseen_dir + '/test/*/*')
                        .map(dp.parse_image if params.database == 'dresden' else fn, 
                            num_parallel_calls=AUTOTUNE 
                            if params.database == 'dresden' else None)  
                        .batch(batch_size)
                        .prefetch(buffer_size=AUTOTUNE))

    jpeg_fn = lambda x: tf.py_function(dp.parse_image, inp=[x, 'jpeg'], Tout=[tf.float32, tf.float32])
    jpeg_dataset = (tf.data.Dataset.list_files(params.patch_dir+ '/test/*/*')
                    .map(functools.partial(dp.parse_image, post_processing='jpeg')
                        if params.database == 'dresden' else jpeg_fn,
                        num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))

    blur_fn = lambda x: tf.py_function(dp.parse_image, inp=[x, 'blur'], Tout=[tf.float32, tf.float32])
    blur_dataset = (tf.data.Dataset.list_files(params.patch_dir+ '/test/*/*')
                    .map(functools.partial(dp.parse_image, post_processing='blur') 
                        if params.database == 'dresden' else blur_fn,
                        num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))
    
    noise_fn = lambda x: tf.py_function(dp.parse_image, inp=[x, 'noise'], Tout=[tf.float32, tf.float32])
    noise_dataset = (tf.data.Dataset.list_files(params.patch_dir+ '/test/*/*')
                    .map(functools.partial(dp.parse_image, post_processing='noise')
                        if params.database == 'dresden' else noise_fn,
                        num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))

    in_iter = iter(in_dataset)
    unseen_iter = iter(unseen_dataset)
    jpeg_iter = iter(jpeg_dataset)
    blur_iter = iter(blur_dataset)
    noise_iter = iter(noise_dataset)

    # In Out Distinction
    # images from unseen models
    
    unseen_ds_size = 0
    for m in params.unseen_brand_models:
        unseen_ds_size += len(os.listdir(os.path.join(params.unseen_dir, 'test', m)))
    num_unseen_steps = (unseen_ds_size + batch_size - 1) // batch_size
    num_test_steps = (test_size + batch_size - 1) // batch_size

    unseen_prob_in, unseen_prob_out = utils.in_out_distinction(in_iter, unseen_iter, 
                                                                model, num_unseen_steps,
                                                                ood_name='UNSEEN MODEL',
                                                                fname=fname)
    # jpeg images
    jpeg_prob_in, jpeg_prob_out = utils.in_out_distinction(in_iter, jpeg_iter, 
                                                            model, num_test_steps,
                                                            ood_name='JPEG',
                                                            fname=fname)
    # blurred images
    blur_prob_in, blur_prob_out = utils.in_out_distinction(in_iter, blur_iter, 
                                                            model, num_test_steps,
                                                            ood_name='BLUR',
                                                            fname=fname)
    # blurred images
    noise_prob_in, noise_prob_out = utils.in_out_distinction(in_iter, noise_iter, 
                                                                model, num_test_steps,
                                                                ood_name='NOISE',
                                                                fname=fname)

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
    canvas.print_figure(stats_fig, format='png')
    print('saved {}'.format(stats_fig))

if __name__ == "__main__":
    ckpt_dir = 'ckpts/' + params.database + '/' + params.model_type
    stats_fig = 'results/' + params.database + '/' + params.model_type + '_stats.png'
    fname = 'results/' + params.database + '/' + params.model_type + '_stats.log'
    download = False
    if params.model_type == 'bnn':
        bnn_stats(ckpt_dir, stats_fig, fname, download)
    elif params.model_type == 'vanilla':
        vanilla_stats(ckpt_dir, stats_fig, fname, download)

