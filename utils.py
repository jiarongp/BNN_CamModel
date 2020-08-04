import logging
import params
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib
import sklearn.metrics as sk
from tqdm import trange
matplotlib.use('Agg')
from matplotlib import figure
from matplotlib.backends import backend_agg


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
    fig = figure.Figure(figsize=(13, 3*n))
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


def compute_probs(model, images, num_monte_carlo=30):
    probs = tf.stack([tf.keras.layers.Activation('softmax')(model(images)) for _ in range(num_monte_carlo)], axis=0)
    mean_probs = tf.reduce_mean(probs, axis=0)
    eps = tf.convert_to_tensor(np.finfo(float).eps, dtype=tf.float32)
    sum_log_prob = -tf.reduce_sum(tf.math.multiply(mean_probs, tf.math.log(mean_probs + eps)), axis=1)
    heldout_log_prob = tf.reduce_mean(sum_log_prob)
    print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))
    return probs, heldout_log_prob


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


# right wrong distinction is for in distribution classification, to see how good the 
# classifier perform in in distribution examples.
def right_wrong_distinction(test_iterator, model, num_test_steps, fname='results/baseline.log'):
    softmax_prob_all, softmax_prob_right, softmax_prob_wrong, accuracy = [], [], [], []
    for step in trange(num_test_steps):
        images, onehot_labels = test_iterator.get_next()
        # the model that I import has softmax already as last layer
        
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
        aupr, auroc = area_under_curves(np.asarray(softmax_prob_right), np.asarray(softmax_prob_wrong))
        f.write('AUPR (%): {}\n'.format(aupr))
        f.write('AUROC (%): {}\n'.format(auroc))
        f.write('[ERROR Detection]\n')
        f.write('Prediction Prob: Right/Wrong classification distinction\n')
        aupr, auroc = area_under_curves(-np.asarray(softmax_prob_right), -np.asarray(softmax_prob_wrong), True)
        f.write('AUPR (%): {}\n'.format(aupr))
        f.write('AUROC (%): {}\n'.format(auroc))

    return softmax_prob_right, softmax_prob_wrong


# in out distinction is for out distribution classifier
def in_out_distinction(in_iter, out_iter, model, num_test_steps, ood_name, fname='results/baseline.log'):
    softmax_prob_in, softmax_prob_out, norm_base_rate, abnorm_base_rate = [], [], [], []
    for step in trange(num_test_steps):
        in_images, _ = in_iter.get_next()
        out_images, _ = out_iter.get_next()
        logits_in = model(in_images)
        logits_out = model(out_images)
        softmax_in = tf.nn.softmax(logits_in)
        softmax_out = tf.nn.softmax(logits_out)
        s_prob_in = np.amax(softmax_in, axis=1, keepdims=True)
        s_prob_out = np.amax(softmax_out, axis=1, keepdims=True)

        norm_base_rate.append(in_images.shape[0] / 
                             (in_images.shape[0] + out_images.shape[0]))
        abnorm_base_rate.append(out_images.shape[0] / 
                               (in_images.shape[0] + out_images.shape[0]))
        softmax_prob_in.extend(s_prob_in)
        softmax_prob_out.extend(s_prob_out)

    with open(fname, 'a') as f:
        f.write("\nIn Out Distinction\n")
        f.write("[{} anomaly detection]\n".format(ood_name))
        f.write('In-dist max softmax distribution (mean, std):\n')
        f.write('{:.4f}, {:.4f}\n'.format(np.mean(softmax_prob_in),
                                         np.std(softmax_prob_in)))

        f.write('Out-of-dist max softmax distribution(mean, std):\n')
        f.write('{:.4f}, {:.4f}\n'.format(np.mean(softmax_prob_out),
                                         np.std(softmax_prob_out)))

        f.write('[Normality Detection]\n')
        f.write('Normality base rate (%): {}\n'.format(100 * round(np.mean(norm_base_rate), 2)))
        f.write('Prediction Prob: Normality Detection\n')
        aupr, auroc = area_under_curves(np.asarray(softmax_prob_in), np.asarray(softmax_prob_out))
        f.write('AUPR (%): {}\n'.format(aupr))
        f.write('AUROC (%): {}\n'.format(auroc))

        f.write('[Abnormality Detection]\n')
        f.write('Abnormality base rate (%): {}\n'.format(100 * round(np.mean(abnorm_base_rate), 2)))
        f.write('Prediction Prob: Abnormality Detection\n')
        aupr, auroc = area_under_curves(-np.asarray(softmax_prob_in), -np.asarray(softmax_prob_out), True)
        f.write('AUPR (%): {}\n'.format(aupr))
        f.write('AUROC (%): {}\n'.format(auroc))

    return softmax_prob_in, softmax_prob_out

def area_under_curves(safe, risky, inverse=False):
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    if inverse:
        labels[safe.shape[0]:] += 1
    else:
        labels[:safe.shape[0]] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    aupr = round(100 * sk.average_precision_score(labels, examples), 2)
    auroc =  round(100 * sk.roc_auc_score(labels, examples), 2)
    return aupr, auroc

def roc_pr_curves(safe, risky, inverse=False):
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    if inverse:
        labels[safe.shape[0]:] += 1
    else:
        labels[:safe.shape[0]] += 1
    examples = np.squeeze(np.vstack((safe, risky)))
    
    auroc = round(100 * sk.roc_auc_score(labels, examples), 2)
    fpr, tpr, _ = sk.roc_curve(labels, examples)
    aupr = round(100 * sk.average_precision_score(labels, examples), 2)
    precision, recall, _ = sk.precision_recall_curve(labels, examples)

    return fpr, tpr, precision, recall, aupr, auroc