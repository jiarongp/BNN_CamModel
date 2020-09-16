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


def stats(ckpt_dir, stats_fig, fname):
    # collect data from unseen models
    dp.collect_unseen()
    dp.collect_kaggle

    # load model
    if params.model_type == 'bnn':
        train_size = 0
        for m in params.brand_models:
            train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
        # import bnn model
        model = model_lib.bnn(train_size)
    else:
        model = model_lib.vanilla()
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']),
        net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    # form dataset and unseen dataset.
    # both dataset may have different size, might need different batch size.
    ds, num_test_batches = dp.aligned_ds(params.patch_dir, 
                                      params.brand_models)
    unseen_ds, num_unseen_batches = dp.aligned_ds(params.unseen_dir, 
                                               params.unseen_brand_models,
                                               num_batches=num_test_batches)
    kaggle_models = os.listdir(os.path.join('data', 'kaggle'))
    kaggle_ds, num_kaggle_batches = dp.aligned_ds(params.kaggle_dir,
                                                  kaggle_models,
                                                  num_batches=num_test_batches)

    in_dataset = (tf.data.Dataset.from_tensor_slices(ds)
                    .repeat()
                    .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                    .batch(params.BATCH_SIZE)
                    .prefetch(buffer_size=AUTOTUNE))
    
    unseen_dataset = (tf.data.Dataset.from_tensor_slices(unseen_ds)
                            .repeat()
                            .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                            .batch(params.BATCH_SIZE)
                            .prefetch(buffer_size=AUTOTUNE))

    kaggle_dataset = (tf.data.Dataset.from_tensor_slices(kaggle_ds)
                            .repeat()
                            .map(dp.parse_image, num_parallel_calls=AUTOTUNE)
                            .batch(params.BATCH_SIZE)
                            .prefetch(buffer_size=AUTOTUNE))

    jpeg_ds = dp.post_processing(ds, 'jpeg', 70)
    jpeg_dataset = (tf.data.Dataset.from_tensor_slices(jpeg_ds)
                    .repeat()
                    .map(dp.parse_image,
                         num_parallel_calls=AUTOTUNE)
                    .batch(params.BATCH_SIZE)
                    .prefetch(buffer_size=AUTOTUNE))

    blur_ds = dp.post_processing(ds, 'blur', 1.1)
    blur_dataset = (tf.data.Dataset.from_tensor_slices(blur_ds)
                    .repeat()
                    .map(dp.parse_image,
                         num_parallel_calls=AUTOTUNE)
                    .batch(params.BATCH_SIZE)
                    .prefetch(buffer_size=AUTOTUNE))
    
    noise_ds = dp.post_processing(ds, 'noise', 2.0)
    noise_dataset = (tf.data.Dataset.from_tensor_slices(noise_ds)
                        .repeat()
                        .map(dp.parse_image,
                             num_parallel_calls=AUTOTUNE)
                        .batch(params.BATCH_SIZE)
                        .prefetch(buffer_size=AUTOTUNE))

    in_iter = iter(in_dataset)
    unseen_iter = iter(unseen_dataset)
    kaggle_iter = iter(kaggle_dataset)
    jpeg_iter = iter(jpeg_dataset)
    blur_iter = iter(blur_dataset)
    noise_iter = iter(noise_dataset)

    if params.model_type == 'vanilla':
        # inverse is to set in-dist as positive label (by default the ood samples has positive)
        # In softmax statistics, in-dist samples has higher softmax probability
        # setting in-dist as positive will make the roc curve looks natural.
        inverse = True
        # Right Wrong Distinction, just to show how good the model is in terms of
        # in-distribution classification.
        test_iter = dp.build_dataset('test', 'dresden')
        softmax_prob_right, softmax_prob_wrong = utils.right_wrong_distinction(
                                                            test_iter, model, 
                                                            num_test_batches,
                                                            fname)
        # in vanilla cnn, the number of monte carlo is set to 1.
        s_p_in = utils.in_stats(in_iter, model, num_test_batches)

        # unseen images
        s_p_unseen, unseen_class_num = utils.out_stats(unseen_iter, 
                                                       model, num_unseen_batches)
        # write logging information to file
        utils.log_in_out(s_p_in, s_p_unseen, unseen_class_num,
                         'UNSEEN MODEL', fname)
        # jpeg images
        s_p_jpeg, jpeg_class_num = utils.out_stats(jpeg_iter, 
                                                   model, num_test_batches)
        utils.log_in_out(s_p_in, s_p_jpeg, jpeg_class_num, 'JPEG', fname)

        # blurred images
        s_p_blur, blur_class_num = utils.out_stats(blur_iter,
                                                   model, num_test_batches)
        utils.log_in_out(s_p_in, s_p_blur, blur_class_num, 'BLUR', fname)

        # noisy images
        s_p_noise, noise_class_num = utils.out_stats(noise_iter,
                                                     model, num_test_batches)
        utils.log_in_out(s_p_in, s_p_noise, noise_class_num, 'NOISE', fname)

        # Bind softmax right/wrong distinction
        s_p_rw = [softmax_prob_right, softmax_prob_wrong]
        s_p_io_unseen = [s_p_in, s_p_unseen]
        s_p_io_jpeg = [s_p_in, s_p_jpeg]
        s_p_io_blur = [s_p_in, s_p_blur]
        s_p_io_noise = [s_p_in, s_p_noise]

        # if the number of unseen batches is smaller or larger, histogram
        # should be scaled to fit the others.
        labels = ['In Distribution', 'Unseen Models', 'JPEG', 'Blurred', 'Noisy']
        data = [s_p_in, s_p_unseen, s_p_jpeg, s_p_blur, s_p_noise]
        softmax_fig = os.path.join('result', params.database, params.model_type) + '_softmax_dist.png'
        utils.vis_softmax_hist(data, labels, softmax_fig, 
                                "Softmax Statistics")

        targets = [('right/wrong', s_p_rw),
                    ('in/out, unseen models', s_p_io_unseen),
                    ('in/out, jpeg', s_p_io_jpeg),
                    ('in/out, blur', s_p_io_blur),
                    ('in/out, noise', s_p_io_noise)]
        

    elif params.model_type == 'bnn':
        inverse = False
        print('number of unseen batches {}'.format(num_unseen_batches))
        print('number of test batches {}'.format(num_test_batches))

        print("... In-distribution MC Statistics")
        in_entropy, in_epistemic = utils.mc_in_stats(in_iter, model,
                                                      num_test_batches, 
                                                      params.num_monte_carlo)

        # unseen images
        print("... UNSEEN out-of-distribution MC Statistics")
        unseen_entropy, unseen_epistemic, unseen_cls_count = \
                        utils.mc_out_stats(unseen_iter, model, 
                                           num_unseen_batches,
                                           params.num_monte_carlo)
        utils.log_mc_in_out(in_entropy,
                            unseen_entropy,
                            in_epistemic,
                            unseen_epistemic,
                            unseen_cls_count,
                            params.num_monte_carlo,
                            'UNSEEN', fname)

        print("... Kaggle out-of-distribution MC Statistics")
        kaggle_entropy, kaggle_epistemic, kaggle_cls_count = \
                        utils.mc_out_stats(kaggle_iter, model, 
                                           num_kaggle_batches,
                                           params.num_monte_carlo)
        utils.log_mc_in_out(in_entropy,
                            kaggle_entropy,
                            in_epistemic,
                            kaggle_epistemic,
                            kaggle_cls_count,
                            params.num_monte_carlo,
                            'Kaggle', fname)

        print("... JPEG out-of-distribution MC Statistics")
        jpeg_entropy, jpeg_epistemic, jpeg_cls_count = \
                        utils.mc_out_stats(jpeg_iter, model, 
                                           num_test_batches,
                                           params.num_monte_carlo)
        utils.log_mc_in_out(in_entropy,
                            jpeg_entropy,
                            in_epistemic,
                            jpeg_epistemic,
                            jpeg_cls_count,
                            params.num_monte_carlo,
                            'JPEG', fname)
        print("... BLUR out-of-distribution MC Statistics")
        blur_entropy, blur_epistemic, blur_cls_count = \
                        utils.mc_out_stats(blur_iter, model, 
                                           num_test_batches,
                                           params.num_monte_carlo)
        utils.log_mc_in_out(in_entropy,
                            blur_entropy,
                            in_epistemic,
                            blur_epistemic,
                            blur_cls_count,
                            params.num_monte_carlo,
                            'BLUR', fname)

        print("... NOISE out-of-distribution MC Statistics")
        noise_entropy, noise_epistemic, noise_cls_count = \
                        utils.mc_out_stats(noise_iter, model, 
                                           num_test_batches,
                                           params.num_monte_carlo)
        utils.log_mc_in_out(in_entropy,
                            noise_entropy,
                            in_epistemic,
                            noise_epistemic,
                            noise_cls_count,
                            params.num_monte_carlo,
                            'NOISE', fname)
        
        labels = ['In Distribution', 'Unseen Models', 
                    'Kaggle', 'JPEG', 'Blurred', 'Noisy']

        entropy = [in_entropy, unseen_entropy, kaggle_entropy,
                   jpeg_entropy, blur_entropy, noise_entropy]
        entropy_fig = os.path.join('results', params.database, 
                                    params.model_type) + '_entropy_dist.png'
        utils.vis_uncertainty_hist(entropy, labels, entropy_fig, 
                       "Entropy")

        epistemic = [in_epistemic, unseen_epistemic, kaggle_epistemic,
                     jpeg_epistemic, blur_epistemic, noise_epistemic]
        epistemic_fig = os.path.join('results', params.database,
                                    params.model_type) + '_epistemic_dist.png'
        utils.vis_uncertainty_hist(epistemic, labels, epistemic_fig, 
                       "Epistemic")

        targets = [('entropy, unseen models', 
                    [in_entropy, unseen_entropy]),
                    ('entropy, kaggle',
                    [in_entropy, kaggle_entropy]),
                    ('entropy, jpeg models', 
                    [in_entropy, jpeg_entropy]), 
                    ('entropy, blur models', 
                    [in_entropy, blur_entropy]),
                    ('entropy, noise models', 
                    [in_entropy, noise_entropy]), 
                    ('epistemic, unseen models', 
                    [in_epistemic, unseen_epistemic]),
                    ('epistemic, kaggle', 
                    [in_epistemic, kaggle_epistemic]),
                    ('epistemic, jpeg models', 
                    [in_epistemic, jpeg_epistemic]),
                    ('epistemic, blur models', 
                    [in_epistemic, blur_epistemic]),
                    ('epistemic, noise models', 
                    [in_epistemic, noise_epistemic])]

    # Plotting ROC and PR curves 
    fig = figure.Figure(figsize=(25, 10) 
                        if params.model_type == 'bnn' 
                        else (25, 5))
    canvas = backend_agg.FigureCanvasAgg(fig)
    fz = 15

    opt_list = []
    sns.set_style("darkgrid")
    for i, (plotname, (safe, risky)) in enumerate(targets):
        if params.model_type == 'bnn':
            ax = fig.add_subplot(2, 5, i+1)
        else:
            ax = fig.add_subplot(1, 5, i+1)

        fpr, tpr, opt, auroc = utils.roc_pr_curves(safe, risky, inverse)
        opt_list.append(opt)
        acc = np.sum((risky > opt[2]).astype(int)) / risky.shape[0]
        msg = (plotname + '\n'
              "false positive rate: {:.3%}, "
              "true positive rate: {:.3%}, "
              "threshold: {}, "
              "acc: {:.3%}\n".format(opt[0], opt[1], opt[2], acc))
        with open(fname, 'a') as f:
            f.write(msg)

        ax.plot(fpr, tpr, '-',
                label='AUROC:{}'.format(auroc),
                lw=2)
        ax.plot([0, 1], 'k-', lw=2, label='Base rate(0.5)')
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
    ckpt_dir = os.path.join('ckpts', params.database, params.model_type)
    stats_fig = os.path.join('results', params.database, params.model_type) + '_stats.png'
    fname = os.path.join('results', params.database, params.model_type) + '_stats.log'
    stats(ckpt_dir, stats_fig, fname)

