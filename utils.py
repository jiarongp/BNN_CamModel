import logging
import params
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib
import sklearn.metrics as sk
from tqdm import trange
matplotlib.use('Agg')
plt = matplotlib.pyplot
from matplotlib import figure
from matplotlib.backends import backend_agg

color = sns.color_palette("Set2")
 

# ---------------------------- Logging ---------------------------------------

def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# ---------------------------- Uncertainty ---------------------------------------

def decompose_uncertainties(p_hat):
    """
    Given a number of draws, decompose the predictive variance into aleatoric and epistemic uncertainties.
    Explanation: https://github.com/ykwon0407/UQ_BNN/issues/3
      T: number of draws from the model
      K: number of classes

    For squashing the resulting matrices into a single scalar, there are multiple options:
      * Sum/Average over all elements can result in negative outcomes.
      * Sum/Average over diagonal elements only.
    :param p_hat: ndarray of shape [num_draws, num_classes]
    :return: aleatoric and epistemic uncertainties, each is an ndarray of shape [num_classes, num_classes]
        The diagonal entries of the epistemic uncertainties matrix represents the variances, i.e., np.var(p_hat, axis=0)).
    """
    num_draws = p_hat.shape[0]
    p_mean = np.mean(p_hat, axis=0)
    # Aleatoric uncertainty: \frac{1}{T} \sum\limits_{t=1}^T diag(\hat{p_t}) - \hat{p_t} \hat{p_t}^T
    # Explanation: Split into two sums.
    # 1. \frac{1}{T} \sum\limits_{t=1}^T diag(\hat{p_t})
    #    This is equal to the diagonal of p_mean.
    # 2. \frac{1}{T} \sum\limits_{t=1}^T - \hat{p_t} \hat{p_t}^T
    #    For each element of the sum this produces an array of shape [num_classes, num_classes]
    #    This can be vectorized with dot(p_hat^T, p_hat), which is [num_classes, num_draws] * [num_draws, num_classes] -> [num_classes, num_classes]
    #    Eventually, we need to divide by T
    aleatoric = np.diag(p_mean) - p_hat.T.dot(p_hat) / num_draws

    # Epistemic uncertainty: \frac{1}{T} \sum\limits_{t=1}^T (\hat{p_t} - \bar{p}) (\hat{p_t} - \bar{p})^T
    tmp = p_hat - p_mean
    epistemic = tmp.T.dot(tmp) / num_draws

    return aleatoric, epistemic


def compute_probs(model, images, num_monte_carlo=30):
    probs = tf.stack([tf.keras.layers.Activation('softmax')(model(images)) for _ in range(num_monte_carlo)], axis=0)
    mean_probs = tf.reduce_mean(probs, axis=0)
    eps = tf.convert_to_tensor(np.finfo(float).eps, dtype=tf.float32)
    sum_log_prob = -tf.reduce_sum(tf.math.multiply(mean_probs, tf.math.log(mean_probs + eps)), axis=1)
    heldout_log_prob = tf.reduce_mean(sum_log_prob)
    print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))
    return probs, heldout_log_prob


def image_uncertainty(mc_s_prob):
    # using entropy based method calculate uncertainty for each image
    # mc_s_prob -> (# mc, # batches * batch_size, # classes)
    # mean over the mc samples (# batches * batch_size, # classes)
    mean_probs = np.mean(mc_s_prob, axis=0)
    # log_prob over classes (# batches * batch_size)
    entropy = -np.sum((mean_probs * np.log(mean_probs + np.finfo(float).eps)), axis=1)

    epistemic_all = []
    for i in range(mc_s_prob.shape[1]): # for each image
        # output epistemic uncertainty for each image -> [# classes, # classes] matrix
        aleatoric, epistemic = decompose_uncertainties(mc_s_prob[:,i,:])
        # summarize the matrix 
        epistemic_all.append(sum(np.diag(epistemic)))
    epistemic_all = np.asarray(epistemic_all)
    return entropy, epistemic_all


# ---------------------------- Visualization ---------------------------------------

def plot_heldout_prediction(images, labels, probs, fname, n=5, title=''):
    """Save a PNG plot visualizing posterior uncertainty on heldout data.
    Args:
    input_vals: A `float`-like Numpy `array` of shape
        `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
        num_heldout, num_classes]` containing Monte Carlo samples of
        class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
    """
    fig = figure.Figure(figsize=(4 * len(params.brand_models), 3*n))
    canvas = backend_agg.FigureCanvasAgg(fig)
    color_list = ['b', 'C1', 'g']
    d2c = dict(zip(params.brand_models, color_list))
    for i in range(n):
        ax = fig.add_subplot(n, 3, 3*i + 1)
        ax.imshow(images[i, :, :, 0], interpolation='None', cmap='gray')
        ax.set_title(labels[i])
        ax.axis('off')
        # probs has shape of ([50, 10000, 10])
        # plot the first ten number in the heldout sequence
        ax = fig.add_subplot(n, 3, 3*i + 2)
        for prob_sample in probs:
            # i indicates the first i-th number in the data
            # just plotting the probabilities output for every prediction
            sns.barplot(np.arange(params.NUM_CLASSES), prob_sample[i, :], alpha=0.1, ax=ax)
            ax.set_ylim([0, 1])
            ax.set_xticklabels(params.brand_models)
        ax.set_title('posterior samples')
        ax.set_xticklabels(params.brand_models, fontdict={'fontsize': 8})
    
        ax = fig.add_subplot(n, 3, 3*i + 3)
        # plot the prediction mean for every test number
        aleatoric, epistemic = decompose_uncertainties(probs.numpy()[:,i,:])
        sns.barplot(np.arange(params.NUM_CLASSES), tf.reduce_mean(probs[:, i, :], axis=0), ax=ax)
        ax.set_ylim([0, 1])
        ax.set_title('epistemic uncertainty\n {}'.format(np.diag(epistemic)))
        ax.set_xticklabels(params.brand_models, fontdict={'fontsize': 8})
    fig.suptitle(title, y=1.05)
    fig.tight_layout()

    canvas.print_figure(fname, format='png')
    print('saved {}'.format(fname))


def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    """Save a PNG plot with histograms of weight means and stddevs.
    Args:
    names: A Python `iterable` of `str` variable names.
    qm_vals: A Python `iterable`, the same length as `names`,
        whose elements are Numpy `array`s, of any shape, containing
        posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
        whose elements are Numpy `array`s, of any shape, containing
        posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
    """
    fig = figure.Figure(figsize=(12, 6))
    canvas = backend_agg.FigureCanvasAgg(fig)

    ax = fig.add_subplot(1, 2, 1)
    for n, qm in zip(names, qm_vals):
        sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
    ax.set_title('weight means')
    ax.set_xlim([-1.5, 1.5])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    for n, qs in zip(names, qs_vals):
        sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
    ax.set_title('weight stddevs')
    ax.set_xlim([0, 1.])

    fig.tight_layout()
    canvas.print_figure(fname, format='png')
    print('saved {}'.format(fname))


# ---------------------------- ROC Curves ---------------------------------------

def area_under_curves(safe, risky, inverse=False):
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    # By default, the ood samples has the positive label
    if inverse:
        labels[:safe.shape[0]] += 1
    else:
        labels[safe.shape[0]:] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    aupr = round(100 * sk.average_precision_score(labels, examples), 2)
    auroc = round(100 * sk.roc_auc_score(labels, examples), 2)
    return aupr, auroc


def roc_pr_curves(safe, risky, inverse=False):
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    if inverse:
        labels[:safe.shape[0]] += 1
    else:
        labels[safe.shape[0]:] += 1
    # examples = np.squeeze(np.vstack((safe, risky)))
    examples = np.concatenate((safe, risky))
    
    auroc = round(100 * sk.roc_auc_score(labels, examples), 2)
    fpr, tpr, thresholds = sk.roc_curve(labels, examples)
    # optimal cutoff is the threshold with (tpr - (1 - fpr)) closest to 0
    distance = np.abs(tpr - (1 - fpr))
    idx = np.argmin(distance)
    opt_thresholds = thresholds[idx]
    opt_fpr = fpr[idx]
    opt_tpr = tpr[idx]
    opt = [opt_fpr, opt_tpr, opt_thresholds]
    # aupr = round(100 * sk.average_precision_score(labels, examples), 2)
    # precision, recall, _ = sk.precision_recall_curve(labels, examples)
    return fpr, tpr, opt, auroc


# ---------------------------- Distinction ---------------------------------------

# right wrong distinction is for in distribution classification, to see how good the 
# classifier perform in in distribution examples.
def right_wrong_distinction(test_iterator, model, num_test_steps, fname):
    softmax_prob_all, softmax_prob_right, softmax_prob_wrong, accuracy = [], [], [], []
    print("... Start right wrong distinction")
    for step in trange(num_test_steps):
        images, onehot_labels = test_iterator.get_next()
        logits = model(images)
        softmax_all = tf.nn.softmax(logits)
        labels = np.argmax(onehot_labels, axis=1)
        
        right_mask = np.equal(np.argmax(softmax_all, axis=1), labels)
        wrong_mask = np.not_equal(np.argmax(softmax_all, axis=1), labels)
        right_all, wrong_all = softmax_all[right_mask], softmax_all[wrong_mask]

        s_prob_all = np.amax(softmax_all, axis=1, keepdims=True)
        s_prob_right = np.amax(right_all, axis=1, keepdims=True)
        s_prob_wrong = np.amax(wrong_all, axis=1, keepdims=True)

        correct_cases = np.equal(np.argmax(softmax_all, axis=1), labels)
        acc = 100 * np.mean(np.float32(correct_cases))

        softmax_prob_all.extend(s_prob_all)
        softmax_prob_right.extend(s_prob_right)
        softmax_prob_wrong.extend(s_prob_wrong)
        accuracy.append(acc)

    accuracy = np.mean(accuracy)
    err = 100 - accuracy

    softmax_prob_right = np.asarray(softmax_prob_right)
    softmax_prob_wrong = np.asarray(softmax_prob_wrong)
    
    with open(fname, 'a') as f:
        f.write("Right Wrong Distinction\n")
        f.write("[SUCCESS DETECTION]\n")
        f.write('Error (%)| Prediction Prob (mean, std) | PProb Right (mean, std) | PProb Wrong (mean, std):\n')
        f.write('{:.4f} | {:.4f}, {:.4f} | {:.4f}, {:.4f} | {:.4f}, {:.4f}\n'.format(
                err, 
                np.mean(softmax_prob_all),
                np.std(softmax_prob_all),
                np.mean(softmax_prob_right),
                np.std(softmax_prob_right),
                np.mean(softmax_prob_wrong),
                np.std(softmax_prob_wrong)))
        f.write('Success base rate (%): {}, ({} / {})\n'.format(
                round(accuracy, 2), len(softmax_prob_right), len(softmax_prob_all)))
        f.write('Prediction Prob: Right/Wrong classification distinction\n')
        aupr, auroc = area_under_curves(softmax_prob_right, softmax_prob_wrong, True)
        f.write('AUPR (%): {}\n'.format(aupr))
        f.write('AUROC (%): {}\n'.format(auroc))
        f.write('[ERROR Detection]\n')
        f.write('Prediction Prob: Right/Wrong classification distinction\n')
        aupr, auroc = area_under_curves(-softmax_prob_right, -softmax_prob_wrong)
        f.write('AUPR (%): {}\n'.format(aupr))
        f.write('AUROC (%): {}\n'.format(auroc))

    return softmax_prob_right, softmax_prob_wrong


def in_stats(in_iter, model, num_test_steps):
    print('... Start computing in distribution statistics')
    softmax_prob_in = []
    for step in trange(num_test_steps):
        in_images, _ = in_iter.get_next()
        logits_in = model(in_images)
        softmax_in = tf.nn.softmax(logits_in)
        s_prob_in = np.amax(softmax_in, axis=1, keepdims=True)
        softmax_prob_in.extend(s_prob_in)
    softmax_prob_in = np.asarray(softmax_prob_in)

    return softmax_prob_in


# in out distinction is for out distribution classifier
def out_stats(out_iter, model, num_test_steps):
    class_count = [0 for m in params.brand_models]
    print('... Start computing out of distribution statistics')
    softmax_prob_out = []
    for step in trange(num_test_steps):
        out_images, _ = out_iter.get_next()
        logits_out = model(out_images)
        softmax_out = tf.nn.softmax(logits_out)
        s_prob_out = np.amax(softmax_out, axis=1, keepdims=True)
        
        for logit in logits_out:
            y_pred = tf.math.argmax(logit)
            class_count[y_pred] += 1
        softmax_prob_out.extend(s_prob_out)
    softmax_prob_out = np.asarray(softmax_prob_out)

    return softmax_prob_out, class_count


def mc_in_stats(in_iter, model, num_test_steps, num_monte_carlo):
    mc_softmax_prob_in = []
    print('... Start computing in distribution statistics')
    for i in trange(num_monte_carlo):
        softmax_prob_in = []
        for step in range(num_test_steps):
            in_images, _ = in_iter.get_next()
            logits_in = model(in_images)
            softmax_in = tf.nn.softmax(logits_in)
            softmax_prob_in.extend(softmax_in)
        mc_softmax_prob_in.append(softmax_prob_in)
    mc_softmax_prob_in = np.asarray(mc_softmax_prob_in)
    entropy, epistemic = image_uncertainty(mc_softmax_prob_in)

    return entropy, epistemic


# in out distinction is for out distribution classifier
def mc_out_stats(out_iter, model, num_test_steps, num_monte_carlo):
    mc_softmax_prob_out = []
    class_count = [0 for m in params.brand_models]
    print('... Start computing out of distribution statistics')
    for i in trange(num_monte_carlo):
        softmax_prob_out = []
        for step in range(num_test_steps):
            out_images, _ = out_iter.get_next()
            logits_out = model(out_images)
            softmax_out = tf.nn.softmax(logits_out)
            for logit in logits_out:
                y_pred = tf.math.argmax(logit)
                class_count[y_pred] += 1
            softmax_prob_out.extend(softmax_out)
        mc_softmax_prob_out.append(softmax_prob_out)
    mc_softmax_prob_out = np.asarray(mc_softmax_prob_out)
    entropy, epistemic = image_uncertainty(mc_softmax_prob_out)

    return entropy, epistemic, class_count


def log_in_out(softmax_prob_in, softmax_prob_out, 
               class_count, ood_name, fname):
    with open(fname, 'a') as f:
        f.write("\nIn Out Distinction\n")
        f.write("[{} anomaly detection]\n".format(ood_name))
        f.write('In-dist max softmax distribution (mean, std):\n')
        f.write('{:.4f}, {:.4f}\n'.format(np.mean(softmax_prob_in),
                                         np.std(softmax_prob_in)))

        f.write('Out-of-dist max softmax distribution(mean, std):\n')
        f.write('{:.4f}, {:.4f}\n'.format(np.mean(softmax_prob_out),
                                         np.std(softmax_prob_out)))

        norm_base_rate = (softmax_prob_in.shape[0] / 
                         (softmax_prob_in.shape[0] + 
                          softmax_prob_out.shape[0]))
        f.write('[Normality Detection]\n')
        f.write('Normality base rate (%): {}\n'.format(100 * round(np.mean(norm_base_rate), 2)))
        f.write('Prediction Prob: Normality Detection\n')
        aupr, auroc = area_under_curves(softmax_prob_in, softmax_prob_out, True)
        f.write('AUPR (%): {}\n'.format(aupr))
        f.write('AUROC (%): {}\n'.format(auroc))

        abnorm_base_rate = (softmax_prob_out.shape[0] / 
                            (softmax_prob_in.shape[0] + 
                            softmax_prob_out.shape[0]))
        f.write('[Abnormality Detection]\n')
        f.write('Abnormality base rate (%): {}\n'.format(100 * round(np.mean(abnorm_base_rate), 2)))
        f.write('Prediction Prob: Abnormality Detection\n')
        aupr, auroc = area_under_curves(-softmax_prob_in, -softmax_prob_out)
        f.write('AUPR (%): {}\n'.format(aupr))
        f.write('AUROC (%): {}\n'.format(auroc))

        total = softmax_prob_out.shape[0]
        f.write("{} out-dist images\n".format(total))
        for i in range(len(params.brand_models)):
            f.write("{:.3%} out-dist images are classified as {}\n".format(
                    class_count[i] / total,
                    params.brand_models[i]))


def log_mc_in_out(in_log_prob, out_log_prob, 
                  in_epistemic, out_epistemic,
                  class_count,
                  num_monte_carlo,
                  ood_name, fname):
    with open(fname, 'a') as f:
        f.write("\n{} In Out Distinction\n".format(ood_name))
        f.write('In-dist log probability (mean, std):\n')
        f.write('{:.4f}, {:.4f}\n'.format(np.mean(in_log_prob),
                                          np.std(in_log_prob)))
        f.write('Out-dist log probability (mean, std):\n')
        f.write('{:.4f}, {:.4f}\n'.format(np.mean(out_log_prob),
                                          np.std(out_log_prob)))
        f.write('In-dist epistemic uncertainty(mean, std):\n')
        f.write('{:.4f}, {:.4f}\n'.format(np.mean(in_epistemic),
                                          np.std(in_epistemic)))
        f.write('Out-dist epistemic uncertainty(mean, std):\n')
        f.write('{:.4f}, {:.4f}\n'.format(np.mean(out_epistemic),
                                          np.std(out_epistemic)))
        total = out_log_prob.shape[0] * num_monte_carlo
        f.write("{} out-dist images\n".format(int(total / num_monte_carlo)))
        for i in range(len(params.brand_models)):
            f.write("{:.3%} out-dist images are classified as {}\n".format(
                    class_count[i] / total,
                    params.brand_models[i]))


# ---------------------------- Histogram ---------------------------------------

def vis_softmax_hist(data, labels, fname, xlabel):
    plt.figure(figsize=(20, 20))
    c = 0
    for d, l in zip(data, labels):
        plt.hist(d, label=l, alpha=0.5, bins=20, color=color[c])
        c += 1
    plt.title("Dataset classification", fontsize=25)
    plt.xlabel("Classification confidence", fontsize=25)
    plt.ylabel("Number of images", fontsize=25)
    plt.legend(loc=0, fontsize=25)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.xlim(left=-0.0, right=1.05)
    plt.savefig(fname, bbox_inches='tight')


def vis_uncertainty_hist(data, labels, fname, xlabel):
    plt.figure(figsize=(20, 20))
    max_uncertainty = []
    c = 0
    for d, l in zip(data, labels):
        max_uncertainty.append(max(d))
        plt.hist(d, label=l, alpha=0.5, bins=20, color=color[c])
        c += 1
    
    plt.title("Dataset classification", fontsize=25)
    plt.xlabel("Classification confidence", fontsize=25)
    plt.ylabel("Number of images", fontsize=25)
    plt.legend(loc=0, fontsize=25)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.xlim(left=-0.0, right=max(max_uncertainty))
    plt.savefig(fname, bbox_inches='tight')