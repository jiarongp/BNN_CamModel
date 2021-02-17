import os
import numpy as np
import tensorflow as tf
import sklearn.metrics as sk
from functools import partial
from tqdm import trange
from utils.misc import write_log
from utils.data_preparation import build_dataset, degradate, parse_image
from utils.visualization import histogram, plot_curve, plot_held_out
keras = tf.keras

class Experiment(object):
    def __init__(self, params):
        self.params = params
        self.random_seed = self.params.experiment.random_seed
        self.in_img_paths, self.num_in_batches = \
                self.aligned_dataset(os.path.join(
                    self.params.dataloader.patch_dir, 'test'),
                    self.params.dataloader.brand_models,
                    seed=self.random_seed)
        self.in_iter = build_dataset(
                        self.params.dataloader.patch_dir,
                        self.params.dataloader.brand_models,
                        "in distribution",
                        self.params.dataloader.batch_size,
                        self.in_img_paths)
        self.log_file = self.params.log.log_file

    def aligned_dataset(self,
            patch_dir, brand_models, batch_size=64,
            num_batches=None, seed=42):
        """
        set a fix subset of total test dataset, so that:
        1. each class has same size of test images (ROC curve
           is sensitive to class imbalance)
        2. each monte carlo draw has the same images as input
        3. random seed controls the how to sample this subset
        Args:
            patch_dir: directory storing patches.
            brand_models: name of camera models for testing.
            batch_size: batch size for testing.
            num_batches: the number of batches from test dataset,
                         use this number to determine the number of 
                         batches of unseen dataset.
            seed: random seed controls the sampling process 
                  from the overall test set.
        """
        # default for 'test' data
        np.random.seed(seed)
        image_paths, num_images, img_paths = [], [], []
        for model in brand_models:
            images = os.listdir(os.path.join(patch_dir, model))
            paths = [os.path.join(patch_dir, model, img) for img in images]
            image_paths.append(paths)
            num_images.append(len(images))
        # number of batches for one class
        class_batches = min(num_images) // batch_size
        num_test_batches = len(brand_models) * class_batches
        # sometimes database has more data in 'test', some has more in 'unseen'
        if num_batches is not None:
            num_test_batches = min(num_test_batches, num_batches)
            class_batches = round(num_test_batches / len(brand_models))
            num_test_batches = len(brand_models) * class_batches
        for images in image_paths:
            np.random.shuffle(images)
            img_paths.extend(images[0:class_batches * batch_size])
        print("class batch number is {}".format(class_batches))
        print("number of test batches is {}".format(num_batches))
        return img_paths, num_test_batches

    def load_checkpoint(self, model, ckpt_dir):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                        optimizer=keras.optimizers.Adam(),
                        net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, 
                        ckpt_dir,
                        max_to_keep=3)
        status = self.ckpt.restore(
                    self.manager.latest_checkpoint).expect_partial()
        # status.assert_existing_objects_matched()
        msg = ("\nRestored from {}\n".format(self.manager.latest_checkpoint))
        write_log(self.log_file, msg)

    def optimal_threshold(self, fpr, tpr, thresholds):
        """
        find out the optimal threshold for the ROC curve, where the optimal 
        cutoff is the threshold with (tpr - (1 - fpr)) closest to 0.
        Args:
            fpr: false positive rate.
            tpr: true positive rate.
            thresholds: the corresponding thresholds.
        Return:
            optimal: the optimal fpr, tpr and threshold for the ROC curve.
        """
        distance = np.abs(tpr - (1 - fpr))
        idx = np.argmin(distance)
        optimal_fpr = fpr[idx]
        optimal_tpr = tpr[idx]
        optimal_thresholds = thresholds[idx]
        optimal = [optimal_fpr, optimal_tpr, optimal_thresholds]
        return optimal

    def roc(self, safe, risky, inverse=False):
        """
        generate roc curve.
        Args:
            safe: examples that are likely to be in-distribution.
            risky: examples that are likely to be out-of-distribution.
            inverse: determine which is case is positive class, by defalut, 
                     the risky examples (out-of-distribution) are positive.
        Return:
            fpr: false positive rates.
            tpr: true positive rates.
            optimal: optimal value for threshold and the corresponding fpr 
                     and tpr.
            auroc: area under the ROC curve.
        """
        labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
        if inverse:
            labels[:safe.shape[0]] += 1
        else:
            labels[safe.shape[0]:] += 1
        examples = np.concatenate((safe, risky))
        auroc = round(100 * sk.roc_auc_score(labels, examples), 2)
        fpr, tpr, thresholds = sk.roc_curve(labels, examples)
        optimal = self.optimal_threshold(fpr, tpr, thresholds)
        # aupr = round(100 * sk.average_precision_score(labels, examples), 2)
        # precision, recall, _ = sk.precision_recall_curve(labels, examples)
        return fpr, tpr, optimal, auroc

    def prepare_unseen_dataset(self):
        # form dataset and unseen dataset.
        # both dataset may have different size, might need different batch size.
        unseen_img_paths, self.num_unseen_batches = \
            self.aligned_dataset(self.params.unseen_dataloader.patch_dir, 
                                self.params.unseen_dataloader.brand_models,
                                num_batches=self.num_in_batches,
                                seed=self.random_seed)
        kaggle_img_paths, self.num_kaggle_batches = \
            self.aligned_dataset(self.params.kaggle_dataloader.patch_dir,
                                self.params.kaggle_dataloader.brand_models,
                                num_batches=self.num_in_batches,
                                seed=self.random_seed)
        self.unseen_iter = build_dataset(
                            self.params.unseen_dataloader.patch_dir,
                            self.params.unseen_dataloader.brand_models,
                            "unseen",
                            self.params.dataloader.batch_size,
                            unseen_img_paths)
        self.kaggle_iter = build_dataset(
                            self.params.kaggle_dataloader.patch_dir,
                            self.params.kaggle_dataloader.brand_models,
                            "kaggle",
                            self.params.dataloader.batch_size,
                            kaggle_img_paths)

    def prepare_degradation_dataset(self, name, factor):
        img_paths = degradate(self.in_img_paths, 
                                self.params.experiment.degradation_dir,
                                self.params.dataloader.database,
                                name, factor)
        patch_dir = os.path.split(img_paths[0])[0]
        iterator = build_dataset(patch_dir,
                                self.params.dataloader.brand_models,
                                name,
                                self.params.dataloader.batch_size,
                                img_paths)
        return iterator

    @tf.function
    def eval_step(self, images):
        logits = self.model(images)
        softmax = tf.nn.softmax(logits)
        max_softmax_cls = tf.one_hot(tf.math.argmax(softmax, axis=1),
                                    len(self.params.dataloader.brand_models))
        return softmax, max_softmax_cls


class SoftmaxStats(Experiment):
    """
    perform softmax statistics.
    """
    def __init__(self, params, model):
        super(SoftmaxStats, self).__init__(params)
        self.model = model
        self.model.build(input_shape=(None, 256, 256, 1))
        self.degradation_id = self.params.softmax_stats.degradation_id
        self.degradation_factor = self.params.softmax_stats.degradation_factor

    def softmax_stats(self, iterator, num_steps):
        """
        compute the maximum class softmax prediction for each example.
        Args:
            iterator: iterator for the target dataset.
            num_steps: num_steps to iterate through the whole dataset.
        Return:
            softmax_prob: maximum class softmax predictions for each example.
            cls_count: the number of predicted outputs for each class.
        """
        softmax_prob = []
        cls_count = [0 for m in self.params.dataloader.brand_models]
        for step in trange(num_steps):
            images, _ = iterator.get_next()
            softmax, max_softmax_cls = self.eval_step(images)
            max_softmax = tf.math.reduce_max(softmax, axis=1, keepdims=True)
            cls_count = [sum(x) for x in zip(tf.math.reduce_sum(
                                                max_softmax_cls, axis=0), 
                                            cls_count)]
            softmax_prob.extend(max_softmax)
        softmax_prob = np.asarray(softmax_prob)
        return softmax_prob, cls_count

    def log_in_out(self, in_softmax_prob, out_softmax_prob, 
                    cls_count, ood_name):
        """
        report the result in log file.
        """
        msg = "{} In Out Distinction\n".format(ood_name)
        msg += 'In-dist max softmax distribution (mean, std):\n'
        msg += ('{:.4f}, {:.4f}\n'.format(np.mean(in_softmax_prob),
                                        np.std(in_softmax_prob)))

        msg += '{} max softmax distribution(mean, std):\n'.format(ood_name)
        msg += ('{:.4f}, {:.4f}\n'.format(np.mean(out_softmax_prob),
                                        np.std(out_softmax_prob)))

        total = out_softmax_prob.shape[0]
        msg += "{} out-dist images\n".format(total)
        for count, brand_model in zip(cls_count, self.params.dataloader.brand_models):
            msg += ("{:.3%} out-dist images are classified as {}\n".format(
                    count / total, brand_model))
        msg += '\n'
        write_log(self.log_file, msg)

    def experiment(self):
        self.load_checkpoint(self.model, 
            self.params.softmax_stats.ckpt_dir)
        self.prepare_unseen_dataset()
        msg = "\n--------------------- Softmax Statistics ---------------------\n\n"
        write_log(self.log_file, msg)

        # In distribution softmax probability
        in_softmax_prob, _ = self.softmax_stats(self.in_iter, 
                                            self.num_in_batches)
        # Unseen images softmax probability
        unseen_softmax_prob, unseen_cls_count = \
            self.softmax_stats(self.unseen_iter,
                                self.num_unseen_batches)
        kaggle_softmax_prob, kaggle_cls_count = \
            self.softmax_stats(self.kaggle_iter,
                                self.num_kaggle_batches)
        # Degradation images softmax probability
        degradation_softmax_prob = []
        degradation_cls_count = []
        degradation_labels = []
        for name, factor in zip(self.degradation_id,
                                self.degradation_factor):
            iterator = self.prepare_degradation_dataset(name, factor)
            softmax_prob, cls_count = \
                self.softmax_stats(iterator, self.num_in_batches)
            degradation_softmax_prob.append(softmax_prob)
            degradation_cls_count.append(cls_count)
            degradation_labels.append(' '.join([name, str(factor)]))

        all_softmax_prob = [in_softmax_prob, unseen_softmax_prob,
                            kaggle_softmax_prob]
        all_softmax_prob.extend(degradation_softmax_prob)
        all_cls_count = [unseen_cls_count, kaggle_cls_count]
        all_cls_count.extend(degradation_cls_count)
        experiment_labels = ["in distribution", "unseen", "kaggle"]
        experiment_labels.extend(degradation_labels)
        for out_softmax_prob, cls_count, label in zip(all_softmax_prob[1:], 
                                                all_cls_count, experiment_labels[1:]):
            self.log_in_out(in_softmax_prob, out_softmax_prob, cls_count, label)
        # plot softmax histograms for each type of data
        histogram(all_softmax_prob,
                experiment_labels,
                "Softmax Statistics Histogram",
                "maximum softmax output from a image",
                self.params.softmax_stats.histogram_path)

        fpr_ls, tpr_ls, auroc_ls = [], [], []
        for out_softmax_prob, plotname in zip(all_softmax_prob[1:], experiment_labels[1:]):
            fpr, tpr, opt_thr, auroc = self.roc(in_softmax_prob, 
                                                out_softmax_prob, 
                                                inverse=True)
            msg = (plotname + '\n'
                "false positive rate: {:.3%}, "
                "true positive rate: {:.3%}, "
                "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
            write_log(self.log_file, msg)
            fpr_ls.append(fpr)
            tpr_ls.append(tpr)
            auroc_ls.append(auroc)
        # plot ROC curve
        plot_curve(experiment_labels[1:],
                [fpr_ls], [tpr_ls], [auroc_ls], 
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                labels=["softmax_stats"],
                suptitle="Softmax Statistics",
                fname=self.params.softmax_stats.roc_path)


class MCStats(Experiment):
    def __init__(self, params, model):
        super(MCStats, self).__init__(params)
        self.model = model
        self.model.build(input_shape=(None, 256, 256, 1))
        self.degradation_id = self.params.mc_stats.degradation_id
        self.degradation_factor = self.params.mc_stats.degradation_factor
        self.num_monte_carlo = self.params.mc_stats.num_monte_carlo
        self.ckpt_dir = self.params.mc_stats.ckpt_dir
        self.entropy_histogram_path = self.params.mc_stats.entropy_histogram_path
        self.epistemic_histogram_path = self.params.mc_stats.epistemic_histogram_path
        self.roc_path =self.params.mc_stats.roc_path

    def decompose_uncertainties(self, p_hat):
        """
        Given a number of draws, decompose the predictive variance into aleatoric and epistemic uncertainties.
        Explanation: https://github.com/ykwon0407/UQ_BNN/issues/3
        T: number of draws from the model
        K: number of classes
        For squashing the resulting matrices into a single scalar, there are multiple options:
        * Sum/Average over all elements can result in negative outcomes.
        * Sum/Average over diagonal elements only.
        Args:
            p_hat: ndarray of shape [num_draws, num_classes]
        Return: 
            aleatoric and epistemic uncertainties, each is an ndarray of shape [num_classes, num_classes]
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

    def image_uncertainty(self, mc_s_prob):
        """
        calculate uncertainty for images.
        """
        # using entropy based method calculate uncertainty for each image
        # mc_s_prob -> (# mc, # batches * batch_size, # classes)
        # mean over the mc samples (# batches * batch_size, # classes)
        mean_probs = np.mean(mc_s_prob, axis=0)
        # log_prob over classes (# batches * batch_size)
        entropy_all = -np.sum((mean_probs * np.log(mean_probs + np.finfo(float).eps)), axis=1)
        # decompose uncertainty
        epistemic_all = []
        for i in range(mc_s_prob.shape[1]): # for each image
            # output epistemic uncertainty for each image -> [# classes, # classes] matrix
            aleatoric, epistemic = self.decompose_uncertainties(mc_s_prob[:,i,:])
            # summarize the matrix 
            epistemic_all.append(sum(np.diag(epistemic)))
        epistemic_all = np.asarray(epistemic_all)
        return entropy_all, epistemic_all

    def mc_stats(self, iterator, num_monte_carlo, num_steps, fname=None):
        """
        compute softmax predictions for each image throughout multiple Monte Carlo samples, 
        and plot the results (optional).
        """
        mc_softmax_prob = []
        cls_count = [0 for m in self.params.dataloader.brand_models]
        for mc_step in trange(num_monte_carlo):
            softmax_prob = []
            for step in range(num_steps):
                images, labels = iterator.get_next()
                softmax, max_softmax_cls = self.eval_step(images)
                cls_count = [sum(x) for x in zip(tf.math.reduce_sum(
                                                    max_softmax_cls, axis=0),
                                                cls_count)]
                softmax_prob.extend(softmax)
            mc_softmax_prob.append(softmax_prob)
        mc_softmax_prob = np.asarray(mc_softmax_prob)
        if fname is not None:
            plot_held_out(images, labels, 
                            self.params.dataloader.brand_models, 
                            mc_softmax_prob[:, 0:64, :], fname)
        return mc_softmax_prob, cls_count

    def log_in_out(self, in_entropy, in_epistemic, 
                    out_entropy, out_epistemic, 
                    cls_count, num_monte_carlo, 
                    ood_name):
        msg = "{} In Out Distinction\n".format(ood_name)
        msg += 'In-dist entropy distribution (mean, std):\n'
        msg += ('{:.4f}, {:.4f}\n'.format(np.mean(in_entropy),
                                        np.std(in_entropy)))
        msg += '{} entropy distribution (mean, std):\n'.format(ood_name)
        msg += ('{:.4f}, {:.4f}\n'.format(np.mean(out_entropy),
                                        np.std(out_entropy)))

        msg += 'In-dist epistemic distribution (mean, std):\n'
        msg += ('{:.4f}, {:.4f}\n'.format(np.mean(in_epistemic),
                                        np.std(in_epistemic)))
        msg += '{} epistemic distribution (mean, std):\n'.format(ood_name)
        msg += ('{:.4f}, {:.4f}\n'.format(np.mean(out_epistemic),
                                        np.std(out_epistemic)))

        total = out_entropy.shape[0] * num_monte_carlo
        msg += "{} out-dist images\n".format(out_entropy.shape[0])
        for count, brand_model in zip(cls_count, self.params.dataloader.brand_models):
            msg += ("{:.3%} out-dist images are classified as {}\n".format(
                    count / total, brand_model))
        msg += '\n'
        write_log(self.log_file, msg)


    def experiment(self):
        self.load_checkpoint(self.model, self.ckpt_dir)
        self.prepare_unseen_dataset()
        msg = "\n--------------------- Monte Carlo Statistics ---------------------\n\n"
        write_log(self.log_file, msg)

        # In distribution probability and uncertainty
        in_mc_s_prob, _ = self.mc_stats(self.in_iter,
                                        self.num_monte_carlo, 
                                        self.num_in_batches,
                                        fname="results/in_distribution.png")
        in_entropy, in_epistemic = self.image_uncertainty(in_mc_s_prob)

        # Unseen images softmax probability and uncertainty
        unseen_mc_s_prob, unseen_cls_count = \
            self.mc_stats(self.unseen_iter,
                            self.num_monte_carlo,
                            self.num_unseen_batches,
                            fname="results/unseen.png")
        unseen_entropy, unseen_epistemic = self.image_uncertainty(unseen_mc_s_prob)
        kaggle_mc_s_prob, kaggle_cls_count = \
            self.mc_stats(self.kaggle_iter,
                            self.num_monte_carlo,
                            self.num_kaggle_batches,
                            fname="results/kaggle.png")
        kaggle_entropy, kaggle_epistemic = self.image_uncertainty(kaggle_mc_s_prob)
        # Degradation images softmax probability and uncertainty
        degradation_entropy = []
        degradation_epistemic = []
        degradation_cls_count = []
        degradation_labels = []
        for name, factor in zip(self.degradation_id,
                                self.degradation_factor):
            iterator = self.prepare_degradation_dataset(name, factor)
            mc_s_prob, cls_count = \
                self.mc_stats(iterator, self.num_monte_carlo, self.num_in_batches,
                                fname="results/{}.png".format(name))
            entropy, epistemic = self.image_uncertainty(mc_s_prob)
            degradation_entropy.append(entropy)
            degradation_epistemic.append(epistemic)
            degradation_cls_count.append(cls_count)
            degradation_labels.append(' '.join([name, str(factor)]))

        all_entropy = [in_entropy, unseen_entropy, kaggle_entropy]
        all_entropy.extend(degradation_entropy)
        all_epistemic = [in_epistemic, unseen_epistemic, kaggle_epistemic]
        all_epistemic.extend(degradation_epistemic)
        all_cls_count = [unseen_cls_count, kaggle_cls_count]
        all_cls_count.extend(degradation_cls_count)
        experiment_labels = ["in distribution", "unseen", "kaggle"]
        experiment_labels.extend(degradation_labels)
        for out_entropy, out_epistemic, cls_count, label in zip(all_entropy[1:], 
                                                                all_epistemic[1:], 
                                                                all_cls_count, 
                                                                experiment_labels[1:]):
            self.log_in_out(in_entropy, in_epistemic,
                            out_entropy, out_epistemic,
                            cls_count, self.num_monte_carlo,
                            label)
        # plot histograms for both entropy-based and epistemic uncertainty
        histogram(all_entropy,
                    experiment_labels,
                    "Monte Carlo Uncertainty(Entropy) Statistics Histogram",
                    "entropy based uncertainty output from an image",
                    self.entropy_histogram_path)
        histogram(all_epistemic,
                    experiment_labels,
                    "Monte Carlo Uncertainty(Epistemic Uncertainty) Statistics Histogram",
                    "epistemic uncertainty output from an image",
                    self.epistemic_histogram_path)

        # generate ROC curves for entropy-based uncertainty
        entropy_fpr, entropy_tpr, entropy_auroc = [], [], []
        for out_entropy, plotname in zip(all_entropy[1:],
                                        experiment_labels[1:]):
            fpr, tpr, opt_thr, auroc = self.roc(in_entropy, out_entropy)
            msg = (plotname + '\n'
                "false positive rate: {:.3%}, "
                "true positive rate: {:.3%}, "
                "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
            write_log(self.log_file, msg)
            entropy_fpr.append(fpr)
            entropy_tpr.append(tpr)
            entropy_auroc.append(auroc)
        # generate ROC curves for epistemic uncertainty
        epistemic_fpr, epistemic_tpr, epistemic_auroc = [], [], []
        for out_epistemic, plotname in zip(all_epistemic[1:],
                                            experiment_labels[1:]):
            fpr, tpr, opt_thr, auroc = self.roc(in_epistemic, out_epistemic)
            msg = (plotname + '\n'
                "false positive rate: {:.3%}, "
                "true positive rate: {:.3%}, "
                "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
            write_log(self.log_file, msg)
            epistemic_fpr.append(fpr)
            epistemic_tpr.append(tpr)
            epistemic_auroc.append(auroc)
        # plot ROC curves
        plot_curve(experiment_labels[1:],
                [entropy_fpr, epistemic_fpr], 
                [entropy_tpr, epistemic_tpr], 
                [entropy_auroc, epistemic_auroc], 
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                labels=["entropy", "epistemic"],
                suptitle="Bayesian Statistics of Uncertainty",
                fname=self.roc_path)

class MultiMCStats(MCStats):
    def __init__(self, params, model):
        super(MultiMCStats, self).__init__(params, model)
        self.num_monte_carlo_ls = self.params.multi_mc_stats.num_monte_carlo_ls 
        self.degradation_id = self.params.multi_mc_stats.degradation_id
        self.degradation_factor = self.params.multi_mc_stats.degradation_factor
        self.ckpt_dir = self.params.multi_mc_stats.ckpt_dir

    def experiment(self):
        self.load_checkpoint(self.model, self.ckpt_dir)
        self.prepare_unseen_dataset()
        msg = "\n--------------------- Multiple Monte Carlo Statistics ---------------------\n\n"
        write_log(self.log_file, msg)

        entropy_fpr_ls, entropy_tpr_ls, entropy_auroc_ls = [], [], []
        epistemic_fpr_ls, epistemic_tpr_ls, epistemic_auroc_ls = [], [], []
        # perform Monte Carlo samples according to the number in the list
        for num_monte_carlo in self.num_monte_carlo_ls:
            # In distribution probability
            in_mc_s_prob, _ = self.mc_stats(self.in_iter,
                                            num_monte_carlo,
                                            self.num_in_batches)
            in_entropy, in_epistemic = self.image_uncertainty(in_mc_s_prob)

            # Unseen images softmax probability
            unseen_mc_s_prob, unseen_cls_count = \
                self.mc_stats(self.unseen_iter,
                                num_monte_carlo,
                                self.num_unseen_batches)
            unseen_entropy, unseen_epistemic = self.image_uncertainty(unseen_mc_s_prob)
            kaggle_mc_s_prob, kaggle_cls_count = \
                self.mc_stats(self.kaggle_iter,
                                num_monte_carlo,
                                self.num_kaggle_batches)
            kaggle_entropy, kaggle_epistemic = self.image_uncertainty(kaggle_mc_s_prob)
            # Degradation images softmax probability
            degradation_entropy = []
            degradation_epistemic = []
            degradation_cls_count = []
            degradation_labels = []
            for name, factor in zip(self.degradation_id,
                                    self.degradation_factor):
                iterator = self.prepare_degradation_dataset(name, factor)
                mc_s_prob, cls_count = \
                    self.mc_stats(iterator, num_monte_carlo, self.num_in_batches)
                entropy, epistemic = self.image_uncertainty(mc_s_prob)
                degradation_entropy.append(entropy)
                degradation_epistemic.append(epistemic)
                degradation_cls_count.append(cls_count)
                degradation_labels.append(' '.join([name, str(factor)]))

            all_entropy = [in_entropy, unseen_entropy, kaggle_entropy]
            all_entropy.extend(degradation_entropy)
            all_epistemic = [in_epistemic, unseen_epistemic, kaggle_epistemic]
            all_epistemic.extend(degradation_epistemic)
            all_cls_count = [unseen_cls_count, kaggle_cls_count]
            all_cls_count.extend(degradation_cls_count)
            experiment_labels = ["in distribution", "unseen", "kaggle"]
            experiment_labels.extend(degradation_labels)
            for out_entropy, out_epistemic, cls_count, label in zip(all_entropy[1:], 
                                                                    all_epistemic[1:], 
                                                                    all_cls_count, 
                                                                    experiment_labels[1:]):
                self.log_in_out(in_entropy, in_epistemic,
                                out_entropy, out_epistemic,
                                cls_count, self.num_monte_carlo,
                                label)

            entropy_fpr, entropy_tpr, entropy_auroc = [], [], []
            for out_entropy, plotname in zip(all_entropy[1:],
                                            experiment_labels[1:]):
                fpr, tpr, opt_thr, auroc = self.roc(in_entropy, out_entropy)
                msg = (plotname + '\n'
                    "false positive rate: {:.3%}, "
                    "true positive rate: {:.3%}, "
                    "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
                write_log(self.log_file, msg)
                entropy_fpr.append(fpr)
                entropy_tpr.append(tpr)
                entropy_auroc.append(auroc)
            entropy_fpr_ls.append(entropy_fpr)
            entropy_tpr_ls.append(entropy_tpr)
            entropy_auroc_ls.append(entropy_auroc)

            epistemic_fpr, epistemic_tpr, epistemic_auroc = [], [], []
            for out_epistemic, plotname in zip(all_epistemic[1:],
                                                experiment_labels[1:]):
                fpr, tpr, opt_thr, auroc = self.roc(in_epistemic, out_epistemic)
                msg = (plotname + '\n'
                    "false positive rate: {:.3%}, "
                    "true positive rate: {:.3%}, "
                    "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
                write_log(self.log_file, msg)
                epistemic_fpr.append(fpr)
                epistemic_tpr.append(tpr)
                epistemic_auroc.append(auroc)
            epistemic_fpr_ls.append(epistemic_fpr)
            epistemic_tpr_ls.append(epistemic_tpr)
            epistemic_auroc_ls.append(epistemic_auroc)

        plot_curve(experiment_labels[1:],
                entropy_fpr_ls, 
                entropy_tpr_ls, 
                entropy_auroc_ls, 
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                labels=self.num_monte_carlo_ls,
                suptitle="Multiple Monte Carlo Experiment",
                fname=self.params.multi_mc_stats.entropy_roc_path)

        plot_curve(experiment_labels[1:],
                epistemic_fpr_ls, 
                epistemic_tpr_ls, 
                epistemic_auroc_ls, 
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                labels=self.num_monte_carlo_ls,
                suptitle="Multiple Monte Carlo Experiment",
                fname=self.params.multi_mc_stats.epistemic_roc_path)

class EnsembleStats(MCStats):
    def __init__(self, params, model):
        super(EnsembleStats, self).__init__(params, model)
        self.degradation_id = self.params.ensemble_stats.degradation_id
        self.degradation_factor = self.params.ensemble_stats.degradation_factor
        self.num_ensemble = self.params.ensemble_stats.num_ensemble

    def ensemble_stats(self, iterator, num_steps):
        softmax_prob = []
        cls_count = [0 for m in self.params.dataloader.brand_models]
        for step in range(num_steps):
            images, _ = iterator.get_next()
            softmax, max_softmax_cls = self.eval_step(images)
            cls_count = [sum(x) for x in zip(tf.math.reduce_sum(
                                                max_softmax_cls, axis=0),
                                            cls_count)]
            softmax_prob.extend(softmax)
        softmax_prob = np.asarray(softmax_prob)
        return softmax_prob, cls_count

    def experiment(self):
        self.prepare_unseen_dataset()
        msg = "\n--------------------- Ensemble Statistics ---------------------\n\n"
        write_log(self.log_file, msg)

        all_ensemble_s_prob = []
        ensemble_cls_count = []
        # run the ensemble of CNNs
        for i in range(self.num_ensemble):
            # run one CNN each time
            single_softmax_prob, single_cls_count = self.single_run(i)
            all_ensemble_s_prob.append(single_softmax_prob)
            ensemble_cls_count.append(single_cls_count)

        all_cls_count = []
        for i in range(len(ensemble_cls_count[0])):
            cls_count = [0 for m in self.params.dataloader.brand_models]
            for c in ensemble_cls_count:
                cls_count = [sum(x) for x in zip(cls_count, c[i])]
            all_cls_count.append(np.array(cls_count)/self.num_ensemble)

        in_entropy, in_epistemic = self.image_uncertainty(
            np.asarray([prob[0] for prob in all_ensemble_s_prob]))
        unseen_entropy, unseen_epistemic = self.image_uncertainty(
            np.asarray([prob[1] for prob in all_ensemble_s_prob]))
        kaggle_entropy, kaggle_epistemic = self.image_uncertainty(
            np.asarray([prob[2] for prob in all_ensemble_s_prob]))

        degradation_entropy = []
        degradation_epistemic = []
        degradation_labels = []
        for i, (name, factor) in enumerate(zip(self.degradation_id,
                                self.degradation_factor)):
            entropy, epistemic = self.image_uncertainty(
                np.asarray([prob[3+i] for prob in all_ensemble_s_prob]))
            degradation_entropy.append(entropy)
            degradation_epistemic.append(epistemic)
            degradation_labels.append(' '.join([name, str(factor)]))

        all_entropy = [in_entropy, unseen_entropy, kaggle_entropy]
        all_entropy.extend(degradation_entropy)
        all_epistemic = [in_epistemic, unseen_epistemic, kaggle_epistemic]
        all_epistemic.extend(degradation_epistemic)
        experiment_labels = ["in distribution", "unseen", "kaggle"]
        experiment_labels.extend(degradation_labels)
        for out_entropy, out_epistemic, cls_count, label in zip(all_entropy[1:], 
                                                                all_epistemic[1:], 
                                                                all_cls_count, 
                                                                experiment_labels[1:]):
            self.log_in_out(in_entropy, in_epistemic,
                            out_entropy, out_epistemic,
                            cls_count, self.num_monte_carlo,
                            label)

        histogram(all_entropy,
                experiment_labels,
                "Monte Carlo Uncertainty(Entropy) Statistics Histogram",
                "entropy based uncertainty output from an image",
                self.params.ensemble_stats.entropy_histogram_path)
        histogram(all_epistemic,
                experiment_labels,
                "Monte Carlo Uncertainty(Epistemic Uncertainty) Statistics Histogram",
                "epistemic uncertainty output from an image",
                self.params.ensemble_stats.epistemic_histogram_path)

        entropy_fpr, entropy_tpr, entropy_auroc = [], [], []
        for out_entropy, plotname in zip(all_entropy[1:],
                                        experiment_labels[1:]):
            fpr, tpr, opt_thr, auroc = self.roc(in_entropy, out_entropy)
            msg = (plotname + '\n'
                "false positive rate: {:.3%}, "
                "true positive rate: {:.3%}, "
                "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
            write_log(self.log_file, msg)
            entropy_fpr.append(fpr)
            entropy_tpr.append(tpr)
            entropy_auroc.append(auroc)

        epistemic_fpr, epistemic_tpr, epistemic_auroc = [], [], []
        for out_epistemic, plotname in zip(all_epistemic[1:],
                                            experiment_labels[1:]):
            fpr, tpr, opt_thr, auroc = self.roc(in_epistemic, out_epistemic)
            msg = (plotname + '\n'
                "false positive rate: {:.3%}, "
                "true positive rate: {:.3%}, "
                "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
            write_log(self.log_file, msg)
            epistemic_fpr.append(fpr)
            epistemic_tpr.append(tpr)
            epistemic_auroc.append(auroc)

        plot_curve(experiment_labels[1:],
                [entropy_fpr, epistemic_fpr], 
                [entropy_tpr, epistemic_tpr], 
                [entropy_auroc, epistemic_auroc], 
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                labels=["entropy", "epistemic"],
                suptitle="Ensemble Experiment",
                fname=self.params.ensemble_stats.roc_path)

    def single_run(self, ensemble_idx):
        """
        run one Vanilla CNN of the ensemble.
        """
        ckpt_dir = os.path.join(self.params.ensemble_stats.ckpt_dir,
                                str(ensemble_idx))
        self.load_checkpoint(self.model, ckpt_dir)
        # In distribution probability
        in_softmax_prob, _ = self.ensemble_stats(self.in_iter,
                                        self.num_in_batches)
        # Unseen images softmax probability
        unseen_softmax_prob, unseen_cls_count = \
            self.ensemble_stats(self.unseen_iter,
                            self.num_unseen_batches)
        kaggle_softmax_prob, kaggle_cls_count = \
            self.ensemble_stats(self.kaggle_iter,
                            self.num_kaggle_batches)
        degradation_softmax_prob = []
        degradation_cls_count = []
        degradation_labels = []
        for name, factor in zip(self.degradation_id,
                                self.degradation_factor):
            iterator = self.prepare_degradation_dataset(name, factor)
            softmax_prob, cls_count = \
                self.ensemble_stats(iterator, self.num_in_batches)
            degradation_softmax_prob.append(softmax_prob)
            degradation_cls_count.append(cls_count)

        all_softmax_prob = [in_softmax_prob, unseen_softmax_prob,
                            kaggle_softmax_prob]
        all_softmax_prob.extend(degradation_softmax_prob)
        all_cls_count = [unseen_cls_count, kaggle_cls_count]
        all_cls_count.extend(degradation_cls_count)
        return all_softmax_prob, all_cls_count


class MCDegradationStats(MCStats):
    def __init__(self, params, model):
        super(MCDegradationStats, self).__init__(params, model)
        self.degradation_id = self.params.mc_degradation_stats.degradation_id
        self.degradation_factor = self.params.mc_degradation_stats.degradation_factor
        self.num_monte_carlo = self.params.mc_degradation_stats.num_monte_carlo
        self.ckpt_dir = self.params.mc_degradation_stats.ckpt_dir
        self.entropy_histogram_path = self.params.mc_degradation_stats.entropy_histogram_path
        self.epistemic_histogram_path = self.params.mc_degradation_stats.epistemic_histogram_path
        self.eval_acc = keras.metrics.CategoricalAccuracy(
                    name='eval_accuracy')

    @tf.function
    def eval_acc_step(self, images, labels):
        """
        evaluate the predictions with accuracy.
        """
        with tf.GradientTape() as tape:
            logits = self.model(images)
        total = tf.math.reduce_sum(labels, axis=0)
        gt = tf.math.argmax(labels, axis=1)
        pred = tf.math.argmax(logits, axis=1)
        corr = labels[pred == gt]
        corr_count = tf.math.reduce_sum(corr, axis=0)
        self.eval_acc.update_state(labels, logits)
        return corr_count, total

    def evaluate(self, test_iter):
        self.eval_acc.reset_states()
        corr_ls = [0 for x in self.params.dataloader.brand_models]
        total_ls = [0 for x in self.params.dataloader.brand_models]
        for step in trange(self.num_in_batches):
            images, labels = test_iter.get_next()
            c, t = self.eval_acc_step(images, labels)
            corr_ls = [sum(x) for x in zip(corr_ls, c)]
            total_ls = [sum(x) for x in zip(total_ls, t)]
        msg ='\n\ntest accuracy: {:.3%}\n'.format(self.eval_acc.result())
        write_log(self.log_file, msg)

        for m, c, t in zip(self.params.dataloader.brand_models, corr_ls, total_ls):
            msg = '{} accuracy: {:.3%}\n'.format(m, c / t)
            write_log(self.log_file, msg)

    def experiment(self):
        self.load_checkpoint(self.model, self.ckpt_dir)
        self.prepare_unseen_dataset()
        msg = "\n--------------------- Degradation Monte Carlo Statistics ---------------------\n\n"
        write_log(self.log_file, msg)

        # In distribution  probability
        in_mc_s_prob, _ = self.mc_stats(self.in_iter,
                                        self.num_monte_carlo,
                                        self.num_in_batches)
        self.evaluate(self.in_iter)
        in_entropy, in_epistemic = self.image_uncertainty(in_mc_s_prob)

        entropy_fpr_ls, entropy_tpr_ls, entropy_auroc_ls = [], [], []
        epistemic_fpr_ls, epistemic_tpr_ls, epistemic_auroc_ls = [], [], []
        for degradation_id, degradation_factor in \
            zip(self.degradation_id, self.degradation_factor):
            # Degradation images softmax probability
            degradation_entropy = []
            degradation_epistemic = []
            degradation_cls_count = []
            degradation_labels = []
            for name, factor in zip(degradation_id, degradation_factor):
                iterator = self.prepare_degradation_dataset(name, factor)
                mc_s_prob, cls_count = \
                    self.mc_stats(iterator, self.num_monte_carlo, self.num_in_batches)
                self.evaluate(iterator)
                msg = ' '.join([name, str(factor)]) + '\n'
                write_log(self.log_file, msg)
                entropy, epistemic = self.image_uncertainty(mc_s_prob)
                degradation_entropy.append(entropy)
                degradation_epistemic.append(epistemic)
                degradation_cls_count.append(cls_count)
                degradation_labels.append(' '.join([name, str(factor)]))

            all_entropy = [in_entropy]
            all_entropy.extend(degradation_entropy)
            all_epistemic = [in_epistemic]
            all_epistemic.extend(degradation_epistemic)
            all_cls_count = degradation_cls_count
            experiment_labels = ["in distribution"]
            experiment_labels.extend(degradation_labels)
            for out_entropy, out_epistemic, cls_count, label in zip(all_entropy[1:], 
                                                                    all_epistemic[1:], 
                                                                    all_cls_count, 
                                                                    experiment_labels[1:]):
                self.log_in_out(in_entropy, in_epistemic,
                                out_entropy, out_epistemic,
                                cls_count, self.num_monte_carlo,
                                label)

            histogram(all_entropy,
                    experiment_labels,
                    "JPEG Uncertainty(Entropy) Histogram",
                    "entropy based uncertainty",
                    self.entropy_histogram_path + '_{}.png'.format(degradation_factor[0]))

            histogram(all_epistemic,
                    experiment_labels,
                    "JPEG Uncertainty(Epistemic Uncertainty) Histogram",
                    "epistemic uncertainty ",
                    self.epistemic_histogram_path + '_{}.png'.format(degradation_factor[0]))

            entropy_fpr, entropy_tpr, entropy_auroc = [], [], []
            for out_entropy, plotname in zip(all_entropy[1:],
                                            experiment_labels[1:]):
                fpr, tpr, opt_thr, auroc = self.roc(in_entropy, out_entropy)
                msg = (plotname + '\n'
                    "false positive rate: {:.3%}, "
                    "true positive rate: {:.3%}, "
                    "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
                write_log(self.log_file, msg)
                entropy_fpr.append(fpr)
                entropy_tpr.append(tpr)
                entropy_auroc.append(auroc)
            entropy_fpr_ls.append(entropy_fpr)
            entropy_tpr_ls.append(entropy_tpr)
            entropy_auroc_ls.append(entropy_auroc)

            epistemic_fpr, epistemic_tpr, epistemic_auroc = [], [], []
            for out_epistemic, plotname in zip(all_epistemic[1:],
                                                experiment_labels[1:]):
                fpr, tpr, opt_thr, auroc = self.roc(in_epistemic, out_epistemic)
                msg = (plotname + '\n'
                    "false positive rate: {:.3%}, "
                    "true positive rate: {:.3%}, "
                    "threshold: {:.5}\n".format(opt_thr[0], opt_thr[1], opt_thr[2]))
                write_log(self.log_file, msg)
                epistemic_fpr.append(fpr)
                epistemic_tpr.append(tpr)
                epistemic_auroc.append(auroc)
            epistemic_fpr_ls.append(epistemic_fpr)
            epistemic_tpr_ls.append(epistemic_tpr)
            epistemic_auroc_ls.append(epistemic_auroc)

        plot_curve(experiment_labels[1:],
                entropy_fpr_ls, 
                entropy_tpr_ls, 
                entropy_auroc_ls, 
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                labels=self.params.mc_degradation_stats.degradation_factor,
                suptitle="Degradation Experiment",
                fname=self.params.mc_degradation_stats.entropy_roc_path)

        plot_curve(experiment_labels[1:],
                epistemic_fpr_ls, 
                epistemic_tpr_ls, 
                epistemic_auroc_ls, 
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                labels=self.params.mc_degradation_stats.degradation_factor,
                suptitle= "Degradation Experiment",
                fname=self.params.mc_degradation_stats.epistemic_roc_path)