import logging
logging.getLogger('tensorflow').disabled = True
import utils
import data_preparation as dp
import os
import params
import numpy as np
from tqdm import trange
import model_lib
import tensorflow as tf
keras = tf.keras
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def evaluate(log, result_dir, ckpt_dir):
    utils.set_logger(log)

    test_size = 0
    train_size = 0
    val_size = []
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
        test_size += len(os.listdir(os.path.join(params.patch_dir, 'test', m)))
        val_size.append(len(os.listdir(os.path.join(params.patch_dir, 'val', m))))
    num_test_steps = (test_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    # Create the input data pipeline
    logging.info("Creating the dataset...")
    test_iterator = dp.build_dataset('test')

    # Define the model
    logging.info("Creating the model...")
    if params.model_type == 'bnn':
        model = model_lib.bnn(train_size)
    else:
        model = model_lib.vanilla()
    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')
    corr_count = [0 for m in params.brand_models]

    if params.model_type == 'bnn':
        # input random tensor to build the model
        random_input = tf.random.normal([1, params.IMG_WIDTH, params.IMG_HEIGHT, 1])
        model(random_input)

        names = [layer.name for layer in model.layers
                if 'flipout' in layer.name]
        # prior distribution of kernel
        pm_vals = [layer.kernel_prior.mean() 
                for layer in model.layers
                if 'flipout' in layer.name]
        ps_vals = [layer.kernel_prior.stddev()
                for layer in model.layers
                if 'flipout' in layer.name] 
        # posterior distribution of kernel
        qm_vals = [layer.kernel_posterior.mean() 
                for layer in model.layers
                if 'flipout' in layer.name]
        qs_vals = [layer.kernel_posterior.stddev() 
                for layer in model.layers
                if 'flipout' in layer.name]
        # posterior distribution of bias
        bm_vals = [layer.bias_posterior.mean()
                for layer in model.layers
                if 'flipout' in layer.name]
        bs_vals = [layer.bias_posterior.stddev() 
                for layer in model.layers
                if 'flipout' in layer.name]

        utils.plot_weight_posteriors(names, pm_vals, ps_vals, 
                                    fname=result_dir + 
                                    "initialized_prior.png")
        utils.plot_weight_posteriors(names, qm_vals, qs_vals, 
                                    fname=result_dir + 
                                    "initialized_weight.png")
        utils.plot_weight_posteriors(names, bm_vals, bs_vals, 
                                    fname=result_dir + 
                                    "initialized_bias.png")

        # loc_mean = [tf.math.reduce_mean(mean).numpy() for mean in pm_vals]
        # loc_std = [tf.math.reduce_std(std).numpy() for std in pm_vals]
        # scale_mean = [tf.math.reduce_mean(mean).numpy() for mean in ps_vals]
        # scale_std = [tf.math.reduce_std(std).numpy() for std in ps_vals]
        # logging.info('prior distribution of kernel:')
        # logging.info('loc mean: {}\nloc std: {}\nscale mean: {}\nscale std: {}'.format(loc_mean, loc_std, scale_mean, scale_std))

        # loc_mean = [tf.math.reduce_mean(mean).numpy() for mean in qm_vals]
        # loc_std = [tf.math.reduce_std(std).numpy() for std in qm_vals]
        # scale_mean = [tf.math.reduce_mean(mean).numpy() for mean in qs_vals]
        # scale_std = [tf.math.reduce_std(std).numpy() for std in qs_vals]
        # logging.info('posterior distribution of kernel:')
        # logging.info('loc mean: {}\nloc std: {}\nscale mean: {}\nscale std: {}'
        #               .format(loc_mean, loc_std, scale_mean, scale_std))

        # loc_mean = [tf.math.reduce_mean(mean).numpy() for mean in bm_vals]
        # loc_std = [tf.math.reduce_std(std).numpy() for std in bm_vals]
        # scale_mean = [tf.math.reduce_mean(mean).numpy() for mean in bs_vals]
        # scale_std = [tf.math.reduce_std(std).numpy() for std in bs_vals]
        # logging.info('posterior distribution of bias')
        # logging.info('loc mean: {}\nloc std: {}\nscale mean: {}\nscale std: {}'.format(loc_mean, loc_std, scale_mean, scale_mean))

    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
    manager = tf.train.CheckpointManager(ckpt, 
                                         ckpt_dir, 
                                         max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    logging.info("Restored from {}".format(manager.latest_checkpoint))

    if params.model_type == 'bnn':
        @tf.function
        def test_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                nll = (keras.losses.
                       categorical_crossentropy(labels,
                                                logits, 
                                                from_logits=True))
                kl = sum(model.losses)
                loss = nll + kl
            total = tf.math.reduce_sum(labels, axis=0).numpy()
            gt = tf.math.argmax(labels, axis=1)
            pred = tf.math.argmax(logits, axis=1)
            corr = labels[pred == gt]
            corr_count = tf.math.reduce_sum(corr, axis=0).numpy()

            test_loss.update_state(loss)
            test_accuracy.update_state(labels, logits)
            return corr_count, total

        corr_ls, total_ls = [[0 for m in params.brand_models] for i in range(2)]
        for step in trange(num_test_steps):
            images, labels = test_iterator.get_next()
            c, t = test_step(images, labels)
            corr_ls += c
            total_ls += t

            if step % params.print_fig_step == 0:
                probs, heldout_log_prob = utils.compute_probs(model, images)
                logging.info(' ... Held-out nats: {:.3f}\n'.format(heldout_log_prob))
                # transform from onehot to strings
                index = tf.argmax(labels.numpy(), axis=1)
                labels = [params.brand_models[i] for i in index]
                utils.plot_heldout_prediction(images, labels, probs,
                                    fname=result_dir + 'step{}_pred.png'.
                                            format(step),
                                    title='mean heldout logprob {:.2f}'
                                    .format(heldout_log_prob))

        pm_vals = [layer.kernel_prior.mean() 
                for layer in model.layers
                if 'flipout' in layer.name]
        ps_vals = [layer.kernel_prior.stddev() 
                for layer in model.layers
                if 'flipout' in layer.name] 
        qm_vals = [layer.kernel_posterior.mean() 
                for layer in model.layers
                if 'flipout' in layer.name]
        qs_vals = [layer.kernel_posterior.stddev() 
                for layer in model.layers
                if 'flipout' in layer.name]
        bm_vals = [layer.bias_posterior.mean() 
                for layer in model.layers
                if 'flipout' in layer.name]
        bs_vals = [layer.bias_posterior.stddev() 
                for layer in model.layers
                if 'flipout' in layer.name]

        utils.plot_weight_posteriors(names, pm_vals, ps_vals, 
                                    fname=result_dir + 
                                    "trained_prior.png")
        utils.plot_weight_posteriors(names, qm_vals, qs_vals, 
                                    fname=result_dir + 
                                    "trained_weight.png")
        utils.plot_weight_posteriors(names, bm_vals, bs_vals, 
                                    fname=result_dir + 
                                    "trained_bias.png")

        # loc_mean = [tf.math.reduce_mean(mean).numpy() for mean in pm_vals]
        # loc_std = [tf.math.reduce_std(std).numpy() for std in pm_vals]
        # scale_mean = [tf.math.reduce_mean(mean).numpy() for mean in ps_vals]
        # scale_std = [tf.math.reduce_std(std).numpy() for std in ps_vals]
        # logging.info('prior distribution of kernel:')
        # logging.info('loc mean: {}\nloc std: {}\nscale mean: {}\nscale std: {}'.format(loc_mean, loc_std, scale_mean, scale_std))

        # loc_mean = [tf.math.reduce_mean(mean).numpy() for mean in qm_vals]
        # loc_std = [tf.math.reduce_std(std).numpy() for std in qm_vals]
        # scale_mean = [tf.math.reduce_mean(mean).numpy() for mean in qs_vals]
        # scale_std = [tf.math.reduce_std(std).numpy() for std in qs_vals]
        # logging.info('posterior distribution of kernel:')
        # logging.info('loc mean: {}\nloc std: {}\nscale mean: {}\nscale std: {}'
        #               .format(loc_mean, loc_std, scale_mean, scale_std))

        # loc_mean = [tf.math.reduce_mean(mean).numpy() for mean in bm_vals]
        # loc_std = [tf.math.reduce_std(std).numpy() for std in bm_vals]
        # scale_mean = [tf.math.reduce_mean(mean).numpy() for mean in bs_vals]
        # scale_std = [tf.math.reduce_std(std).numpy() for std in bs_vals]
        # logging.info('posterior distribution of bias')
        # logging.info('loc mean: {}\nloc std: {}\nscale mean: {}\nscale std: {}'.format(loc_mean, loc_std, scale_mean, scale_mean))
    
    else:
        @tf.function
        def test_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = keras.losses.categorical_crossentropy(labels,
                                                            logits, 
                                                            from_logits=True)
                softmax = tf.nn.softmax(logits)
            # number of samples for each class
            total = tf.math.reduce_sum(labels, axis=0).numpy()
            gt = tf.math.argmax(labels, axis=1)
            pred = tf.math.argmax(logits, axis=1)
            corr = labels[pred == gt]
            corr_count = tf.math.reduce_sum(corr, axis=0).numpy()

            test_loss.update_state(loss)  
            test_accuracy.update_state(labels, logits)
            return corr_count, total


        corr_ls, total_ls = [[0 for m in params.brand_models] for i in range(2)]
        for step in trange(num_test_steps):
            images, labels = test_iterator.get_next()
            c, t = test_step(images, labels)
            corr_ls += c
            total_ls += t

    logging.info('test loss: {:.3f}, test accuracy: {:.3%}\n'.format(test_loss.result(),
                                                        test_accuracy.result()))
    n = 0
    for m, c, t in zip(params.models, corr_ls, total_ls):
        logging.info('{} accuracy: {:.3%}'.format(m, c / t))


if __name__ == '__main__':
    ckpt_dir = 'ckpts/' + params.database + '/' + params.model_type
    result_dir = 'results/' + params.database + '/'
    log = result_dir + params.model_type + '.log'
    evaluate(log, result_dir, ckpt_dir)