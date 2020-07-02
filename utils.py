import logging
import params
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib
import pandas as pd
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
    
def compute_probs(model, images, num_monte_carlo=30):
    probs = tf.stack([tf.keras.layers.Activation('softmax')(model(images)) for _ in range(num_monte_carlo)], axis=0)
    mean_probs = tf.reduce_mean(probs, axis=0)
    eps = tf.convert_to_tensor(np.finfo(float).eps, dtype=tf.float32)
    sum_log_prob = -tf.reduce_sum(tf.math.multiply(mean_probs, tf.math.log(mean_probs + eps)), axis=1)
    heldout_log_prob = tf.reduce_mean(sum_log_prob)
    print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))
    return probs, heldout_log_prob