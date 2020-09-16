import numpy as np
import tensorflow as tf
import params
import data_preparation as dp
import utils
import os
import model_lib
# import pandas as pd
import seaborn as sns
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure
from matplotlib.backends import backend_agg
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def ensemble_stats(iter, model, num_test_batches):
    class_count = [0 for m in params.brand_models]
    softmax_prob = []
    for step in range(num_test_batches):
        images, _ = iter.get_next()
        logits = model(images)
        softmax = tf.nn.softmax(logits)
        for logit in logits:
            y_pred = tf.math.argmax(logit)
            class_count[y_pred] += 1
        softmax_prob.extend(softmax)
    softmax_prob = np.asarray(softmax_prob)
    return softmax_prob, class_count


def stats(ckpt_dir, stats_fig, fname):
    dp.collect_unseen()
    dp.collect_kaggle()

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

    noise_ds = dp.post_processing(ds, 'noise', 0.1)
    noise_dataset = (tf.data.Dataset.from_tensor_slices(noise_ds)
                        .repeat()
                        .map(dp.parse_image,
                             num_parallel_calls=AUTOTUNE)
                        .batch(params.BATCH_SIZE)
                        .prefetch(buffer_size=AUTOTUNE))

    # df_ds = pd.DataFrame(data={'in distribution': ds, 
    #                             'unseen': unseen_ds,
    #                             'jpeg': jpeg_ds,
    #                             'blur': blur_ds,
    #                             'noise': noise_ds})
    # df_ds.to_csv('ds.txt', index=False)

    in_iter = iter(in_dataset)
    unseen_iter = iter(unseen_dataset)
    kaggle_iter = iter(kaggle_dataset)
    jpeg_iter = iter(jpeg_dataset)
    blur_iter = iter(blur_dataset)
    noise_iter = iter(noise_dataset)

    num_model = 10
    ensemble_in = []
    ensemble_unseen = []
    ensemble_kaggle = []
    ensemble_jpeg = []
    ensemble_blur = []
    ensemble_noise = []

    cls_count_in = [0 for m in params.brand_models]
    cls_count_unseen = [0 for m in params.brand_models]
    cls_count_kaggle = [0 for m in params.brand_models]
    cls_count_jpeg = [0 for m in params.brand_models]
    cls_count_blur = [0 for m in params.brand_models]
    cls_count_noise = [0 for m in params.brand_models]
    for n in trange(num_model):
        model = model_lib.vanilla()
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), 
            optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']),
            net=model)
        manager = tf.train.CheckpointManager(ckpt, 
                                            ckpt_dir+'_{}'.format(n),
                                            max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)

        s_p_in, tmp = ensemble_stats(in_iter, model,
                            num_test_batches)
        cls_count_in = [sum(x) for x in zip(tmp, cls_count_in)]

        s_p_unseen, tmp = ensemble_stats(unseen_iter,
                                model, num_unseen_batches)
        cls_count_unseen = [sum(x) for x in zip(tmp, cls_count_unseen)]

        s_p_kaggle, tmp = ensemble_stats(kaggle_iter,
                                model, num_kaggle_batches)
        cls_count_kaggle = [sum(x) for x in zip (tmp, cls_count_kaggle)]

        s_p_jpeg, tmp = ensemble_stats(jpeg_iter,
                            model, num_test_batches)
        cls_count_jpeg = [sum(x) for x in zip(tmp, cls_count_jpeg)]

        s_p_blur, tmp = ensemble_stats(blur_iter,
                            model, num_test_batches)
        cls_count_blur = [sum(x) for x in zip(tmp, cls_count_blur)]

        s_p_noise, tmp = ensemble_stats(noise_iter,
                            model, num_test_batches)
        cls_count_noise = [sum(x) for x in zip(tmp, cls_count_noise)]

        ensemble_in.append(s_p_in)
        ensemble_unseen.append(s_p_unseen)
        ensemble_kaggle.append(s_p_kaggle)
        ensemble_jpeg.append(s_p_jpeg)
        ensemble_blur.append(s_p_blur)
        ensemble_noise.append(s_p_noise)

    ensemble_in = np.asarray(ensemble_in)
    ensemble_unseen = np.asarray(ensemble_unseen)
    ensemble_kaggle = np.asarray(ensemble_kaggle)
    ensemble_jpeg = np.asarray(ensemble_jpeg)
    ensemble_blur = np.asarray(ensemble_blur)
    ensemble_noise = np.asarray(ensemble_noise)

    in_entropy, in_epistemic = \
        utils.image_uncertainty(ensemble_in)

    unseen_entropy, unseen_epistemic = \
        utils.image_uncertainty(ensemble_unseen)
    utils.log_mc_in_out(in_entropy, 
                    unseen_entropy,
                    in_epistemic,
                    unseen_epistemic,
                    cls_count_unseen,
                    num_model,
                    'UNSEEN', fname)

    kaggle_entropy, kaggle_epistemic = \
        utils.image_uncertainty(ensemble_kaggle)
    utils.log_mc_in_out(in_entropy,
                        kaggle_entropy,
                        in_epistemic,
                        kaggle_epistemic,
                        cls_count_kaggle,
                        num_model,
                        'Kaggle', fname)

    jpeg_entropy, jpeg_epistemic = \
        utils.image_uncertainty(ensemble_jpeg)
    utils.log_mc_in_out(in_entropy, 
                    jpeg_entropy,
                    in_epistemic, 
                    jpeg_epistemic,
                    cls_count_jpeg,
                    num_model,
                    'JPEG', fname)

    blur_entropy, blur_epistemic = \
        utils.image_uncertainty(ensemble_blur)
    utils.log_mc_in_out(in_entropy, 
                    blur_entropy,
                    in_epistemic,
                    blur_epistemic,
                    cls_count_blur,
                    num_model,
                    'BLUR', fname)

    noise_entropy, noise_epistemic = \
        utils.image_uncertainty(ensemble_noise)
    utils.log_mc_in_out(in_entropy, 
                    noise_entropy,
                    in_epistemic,
                    noise_epistemic,
                    cls_count_noise,
                    num_model,
                    'NOISE', fname)

    labels = ['In Distribution', 'Unseen Models', 
            'Kaggle', 'JPEG', 'Blurred', 'Noisy']

    entropy = [in_entropy, unseen_entropy, kaggle_entropy,
            jpeg_entropy, blur_entropy, noise_entropy]
    entropy_fig = os.path.join('results', params.database, 
                                'ensemble') + '_entropy_dist.png'
    utils.vis_uncertainty_hist(entropy, labels, entropy_fig, 
                    "Entropy")

    epistemic = [in_epistemic, unseen_epistemic, kaggle_epistemic,
                    jpeg_epistemic, blur_epistemic, noise_epistemic]
    epistemic_fig = os.path.join('results', params.database,
                                'ensemble') + '_epistemic_dist.png'
    utils.vis_uncertainty_hist(epistemic, labels, epistemic_fig, 
                    "Epistemic")

    targets = [('entropy, unseen', 
                [in_entropy, unseen_entropy]),
                ('entropy, kaggle',
                [in_entropy, kaggle_entropy]),
                ('entropy, jpeg', 
                [in_entropy, jpeg_entropy]), 
                ('entropy, blur', 
                [in_entropy, blur_entropy]),
                ('entropy, noise', 
                [in_entropy, noise_entropy]), 
                ('epistemic, unseen', 
                [in_epistemic, unseen_epistemic]),
                ('epistemic, kaggle', 
                [in_epistemic, kaggle_epistemic]),
                ('epistemic, jpeg', 
                [in_epistemic, jpeg_epistemic]),
                ('epistemic, blur', 
                [in_epistemic, blur_epistemic]),
                ('epistemic, noise', 
                [in_epistemic, noise_epistemic])]

    # df_entropy = pd.DataFrame(data={'in distribution': in_entropy, 
    #                             'unseen': unseen_entropy,
    #                             'jpeg': jpeg_entropy,
    #                             'blur': blur_entropy,
    #                             'noise': noise_entropy})
    # df_entropy.to_csv('entropy.txt', index=False)

    # df_epistemic = pd.DataFrame(data={'in distribution': in_epistemic, 
    #                             'unseen': unseen_epistemic,
    #                             'jpeg': jpeg_epistemic,
    #                             'blur': blur_epistemic,
    #                             'noise': noise_epistemic})
    # df_epistemic.to_csv('epistemic.txt', index=False)
    inverse = False
    # Plotting ROC and PR curves 
    fig = figure.Figure(figsize=(25, 10))
    canvas = backend_agg.FigureCanvasAgg(fig)
    fz = 15

    opt_list = []
    sns.set_style("darkgrid")
    for i, (plotname, (safe, risky)) in enumerate(targets):
        ax = fig.add_subplot(2, 5, i+1)

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
    ckpt_dir = os.path.join('ckpts', params.database, 'vanilla')
    stats_fig = os.path.join('results', params.database, 'ensemble') + '_stats.png'
    fname = os.path.join('results', params.database, 'ensemble') + '_stats.log'
    stats(ckpt_dir, stats_fig, fname)






