import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tikzplotlib
import pandas as pd
color_palette = sns.color_palette()
fz = 20

def histogram(data, labels, title, xlabel, fname):
    """
    plot histograms.
    Args:
        data: input data for each label, shape of [# of experiment, # of examples].
        labels: experiment labels.
        title: title of histogram.
        xlabel: x axis's label.
        fnmae: output file path.
    """
    plt.figure(figsize=(20, 20))
    sns.set()
    for i, (d, l) in enumerate(zip(data, labels)):
        plt.hist(d, label=l, color=color_palette[i] ,bins=20, alpha=0.3)
    plt.title(title, fontsize=fz)
    plt.xlabel(xlabel, fontsize=fz)
    plt.ylabel("Number of images", fontsize=fz)
    plt.legend(loc=0, fontsize=fz)
    plt.tick_params(axis='both', which='minor', labelsize=fz)
    # plt.xlim(left=-0.0, right=1.05)
    tikzplotlib.save(fname + ".tex", standalone=True)
    plt.savefig(fname, bbox_inches='tight')
    print("image is saved to {}".format(fname))

def plot_curve(plotname,
            x_list, y_list, area_list,
            xlabel, ylabel, labels,
            suptitle, fname):
    """
    plot ROC curve.
    Args:
        plotname: experiment label for each curve.
        x_list: false positive rate (fpr) for ROC curve.
        y_list: true positive rate (tpr) for ROC curve.
        area_list: list of area under ROC curve.
        xlabel: name for x axis.
        ylabel: name for y axis.
        labels: label shown in the legend.
        suptitle: subtitle for each subplot.
        fname: output file path.
    """
    cols = len(plotname)
    rows = 1
    fig = plt.figure(figsize=(5*cols, 5*rows))
    sns.set()
    for c, (x_ls, y_ls, area_ls, label) in enumerate(zip(x_list, y_list, area_list, labels)):
        for i, (name, x, y, area) in enumerate(zip(plotname, x_ls, y_ls, area_ls)):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.plot(x, y, '-',
                    label='{} AUROC:{}'.format(label, area),
                    lw=3, color=color_palette[c])
            ax.plot([0, 1], 'k-', lw=3)
            ax.legend(fontsize=10)
            ax.set_title(name, fontsize=15)
            ax.set_xlabel(xlabel, fontsize=15)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=15)
    ax.grid(True)
    fig.suptitle(suptitle)
    tikzplotlib.save(fname + ".tex", standalone=True)
    fig.savefig(fname, bbox_inches='tight')
    print("image is saved to {}".format(fname))

def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
    """
    save a PNG plot with histograms of weight means and stddevs.
    Args:
        names: a Python `iterable` of `str` variable names.
        qm_vals: a Python `iterable`, the same length as `names`,
                whose elements are Numpy arrays, of any shape, containing
                posterior means of weight varibles.
        qs_vals: a Python `iterable`, the same length as `names`,
                whose elements are Numpy `array`s, of any shape, containing
                posterior standard deviations of weight varibles.
        fname: Python `str` filename to save the plot to.
    """
    fig = plt.figure(figsize=(12, 6))
    sns.set()

    ax = fig.add_subplot(1, 2, 1)
    for c, (n, qm) in enumerate(zip(names, qm_vals)):
        sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
        # sns.histplot(tf.reshape(qm, shape=[-1]), color=color_palette[c], ax=ax, label=n, 
        #                         stat="density", kde=True, binrange=(0, 0.1))
    ax.set_title('weight means')
    ax.set_xlim([-1.5, 1.5])
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    for c, (n, qs) in enumerate(zip(names, qs_vals)):
        sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
        # sns.histplot(tf.reshape(qs, shape=[-1]), color=color_palette[c], ax=ax, label=n, 
        #                         stat="density", kde=True, binrange=(0, 0.1))
    ax.set_title('weight stddevs')
    ax.set_xlim([0, 1.])
    fig.tight_layout()
    ax.grid(True)
    tikzplotlib.save(fname + ".tex", standalone=True)
    fig.savefig(fname, bbox_inches='tight')
    print("image is saved to {}".format(fname))

def decompose_uncertainties(p_hat):
    """
    decompose uncertainty into aleratoric and epistemic uncertainty.
    same function defined in class MCStats of experiment_lib.py.
    """
    num_draws = p_hat.shape[0]
    p_mean = np.mean(p_hat, axis=0)
    aleatoric = np.diag(p_mean) - p_hat.T.dot(p_hat) / num_draws
    tmp = p_hat - p_mean
    epistemic = tmp.T.dot(tmp) / num_draws
    return aleatoric, epistemic

def image_uncertainty(mc_s_prob):
    """
    compute uncertainty for images.
    """
    # using entropy based method calculate uncertainty for each image
    # mc_s_prob -> (# mc, # batches * batch_size, # classes)
    # mean over the mc samples (# batches * batch_size, # classes)
    mean_probs = np.mean(mc_s_prob, axis=0)
    # log_prob over classes (# batches * batch_size)
    entropy_all = -np.sum((mean_probs * np.log(mean_probs + np.finfo(float).eps)), axis=1)
    epistemic_all = []
    for i in range(mc_s_prob.shape[1]): # for each image
        # output epistemic uncertainty for each image -> [# classes, # classes] matrix
        aleatoric, epistemic = decompose_uncertainties(mc_s_prob[:,i,:])
        # summarize the matrix 
        epistemic_all.append(sum(np.diag(epistemic)))
    epistemic_all = np.asarray(epistemic_all)
    return entropy_all, epistemic_all

def plot_held_out(images, labels, brand_models, mc_softmax_prob, fname):
    """
    plot result examples with Monte Carlo samples.
    """
    entropy, epistemic = image_uncertainty(mc_softmax_prob)
    num_images = len(entropy)
    num_dis_imgs = num_images // 10
    num_classes = len(brand_models)
    fig = plt.figure(figsize=(20, 3*num_dis_imgs))
    d2c = dict(zip(brand_models, color_palette))
    sns.set()
    for i in range(num_dis_imgs):
        idx = i * 10
        ax = fig.add_subplot(num_dis_imgs, 3, 3*i + 1)
        ax.imshow(images[idx, :, :, 0], interpolation='None', cmap='gray')
        # ax.set_title(brand_models[np.argmax(labels.numpy()[idx])])
        ax.axis('off')

        ax = fig.add_subplot(num_dis_imgs, 3, 3*i + 2)
        for prob_sample in mc_softmax_prob:
            sns.barplot(np.arange(num_classes), prob_sample[idx, :], alpha=0.1, ax=ax)
            ax.set_ylim([0, 1])
            ax.set_xticklabels(brand_models, fontdict={'fontsize':7})
        ax.set_title("entropy: {:.3f}".format(entropy[idx]))

        ax = fig.add_subplot(num_dis_imgs, 3, 3*i + 3)
        df = pd.DataFrame(mc_softmax_prob[:,idx,:], columns=brand_models)
        ax = df.mean(axis=0).plot(kind='bar', color=map(d2c.get, df.columns), 
                                    yerr=df.std(axis=0), rot=0, capsize=5)
        ax.set_ylim([0, 1])
        ax.set_xticklabels(brand_models, fontdict={'fontsize': 8})
        ax.set_title("entropy: {:.3f}".format(epistemic[idx]))

    fig.suptitle('Held-out nats: {:.3f}\n'
                'mean epistemic uncertainty: {:.3f}'.format(np.mean(entropy), np.mean(epistemic), y=1.1))
    fig.tight_layout()
    tikzplotlib.save(fname + ".tex", standalone=True)
    fig.savefig(fname, bbox_inches='tight')