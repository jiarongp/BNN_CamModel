import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import os
import params
import data_preparation as dp
import model_lib
import utils
import datetime
from tqdm import trange
keras = tf.keras
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def build_and_train(log, tb_log, ckpt_dir):
    # Set the logger

    utils.set_logger(log)
    logging.info("Creating the datasets...")
    dp.collect_split_extract(parent_dir=params.patch_dir)

    train_size = 0
    val_size = []
    for m in params.brand_models:
        train_size += len(os.listdir(os.path.join(params.patch_dir, 'train', m)))
        val_size.append(len(os.listdir(os.path.join(params.patch_dir, 'val', m))))
    num_train_steps = (train_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE
    num_val_steps = (sum(val_size) + params.BATCH_SIZE - 1) // params.BATCH_SIZE

    # variables for early stopping and saving best models   
    # random guessing
    best_acc = 1.0 / len(params.brand_models)
    best_loss = 10000
    stop_count = 0
    
    # choose class_imbalance here,
    # using oversampling the minority class, 
    # not sure if it really affect the performance, 
    # since still using the whole training size to calculate the steps
    # resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE) in tensorflow example
    class_imbalance = False if params.even_database else True
    train_iterator = dp.build_dataset('train', class_imbalance=class_imbalance)
    val_iterator = dp.build_dataset('val')

    if params.model_type == 'bnn':
        model = model_lib.bnn(train_size)
    else:
        model = model_lib.vanilla()

    # # focal loss produce more guaranteed a quicker converge for training
    # def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
    #     """
    #     focal loss for multi-classification
    #     FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    #     Notice: logits is probability after softmax
    #     gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    #     d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    #     Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    #     Focal Loss for Dense Object Detection, 130(4), 485–491.
    #     https://doi.org/10.1016/j.ajodo.2005.02.022
    #     :param labels: ground truth labels, shape of [batch_size]
    #     :param logits: model's output, shape of [batch_size, num_cls]
    #     :param gamma:
    #     :param alpha:
    #     :return: shape of [batch_size]
    #     """
    #     epsilon = 1.e-9
    #     softmax = tf.nn.softmax(logits)
    #     num_cls = softmax.shape[1]

    #     model_out = tf.math.add(softmax, epsilon)
    #     ce = tf.math.multiply(labels, -tf.math.log(model_out))
    #     weight = tf.math.multiply(labels, tf.math.pow(tf.math.subtract(1., model_out), gamma))
    #     fl = tf.math.multiply(alpha, tf.math.multiply(weight, ce))
    #     reduced_fl = tf.math.reduce_sum(fl, axis=1)
    #     # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    #     return reduced_fl

    loss_object = keras.losses.CategoricalCrossentropy(from_logits=True)
    # loss_object = focal_loss
    # optimizer = keras.optimizers.Adam(lr=params.HParams['init_learning_rate'])
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        params.HParams['init_learning_rate'],
        decay_steps=num_train_steps,
        decay_rate=0.96,
        staircase=True)
    optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)

    train_loss = keras.metrics.Mean(name='train_loss')
    train_acc = keras.metrics.CategoricalAccuracy(name='train_accuracy')
    if params.model_type == 'bnn':
        kl_loss = keras.metrics.Mean(name='kl_loss')
        nll_loss = keras.metrics.Mean(name='nll_loss')
    val_loss = keras.metrics.Mean(name='test_loss')
    val_acc = keras.metrics.CategoricalAccuracy(name='test_accuracy')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = tb_log + current_time + '/train'
    val_log_dir = tb_log + current_time + '/val'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    val_writer = tf.summary.create_file_writer(val_log_dir)

    # save model to a checkpoint
    ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=optimizer,
            net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    if params.restore:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            logging.info("Restored from {}".format(manager.latest_checkpoint))
        else:
            logging.info("Initializing from scratch.")


    if params.model_type == 'bnn':
        
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                nll = loss_object(labels, logits)
                kl = sum(model.losses)
                loss = nll + kl

            gradients = tape.gradient(loss, model.trainable_weights)
            if step % 50 == 0:
                with train_writer.as_default():
                    for grad, t_w in zip(gradients, model.trainable_weights):
                        if 'kernel' in t_w.name:
                            tf.summary.histogram(t_w.name, grad, offset+step)
                    for l in model.layers[1:]:
                        tf.summary.scalar(l.name,  l.losses[0], step=offset+step)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            with train_writer.as_default():
                tf.summary.scalar('learning_rate',
                                optimizer._decayed_lr(tf.float32),
                                step=offset+step)
            kl_loss.update_state(kl)  
            nll_loss.update_state(nll)
            train_loss.update_state(loss)
            train_acc.update_state(labels, logits)

        @tf.function
        def val_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                nll = loss_object(labels, logits)
                kl = sum(model.losses)
                loss = nll + kl
            # number of samples for each class
            total = tf.math.reduce_sum(labels, axis=0).numpy()
            gt = tf.math.argmax(labels, axis=1)
            pred = tf.math.argmax(logits, axis=1)
            corr = labels[pred == gt]
            corr_count = tf.math.reduce_sum(corr, axis=0).numpy()

            val_loss.update_state(loss)
            val_acc.update_state(labels, logits)
            return corr_count, total

    else:
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_object(labels, logits)
            gradients = tape.gradient(loss, model.trainable_weights)
            if step % 100 == 0:
                with train_writer.as_default():
                    for grad, t_w in zip(gradients, model.trainable_weights):
                        if 'kernel' in t_w.name:
                            tf.summary.histogram(t_w.name, grad, offset+step)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            train_loss.update_state(loss)  
            train_acc.update_state(labels, logits)

        @tf.function
        def val_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_object(labels, logits)

            # number of samples for each class
            total = tf.math.reduce_sum(labels, axis=0).numpy()
            gt = tf.math.argmax(labels, axis=1)
            pred = tf.math.argmax(logits, axis=1)
            corr = labels[pred == gt]
            corr_count = tf.math.reduce_sum(corr, axis=0).numpy()

            val_loss.update_state(loss)
            val_acc.update_state(labels, logits)
            return corr_count, total
            
    logging.info('... Training convolutional neural network\n')
    for epoch in range(params.NUM_EPOCHS):
        offset = epoch * num_train_steps
        val_loss.reset_states()
        val_acc.reset_states()
        train_loss.reset_states()
        train_acc.reset_states()
        if params.model_type == 'bnn':
            kl_loss.reset_states()
            nll_loss.reset_states()

        for step in trange(num_train_steps):
            images, labels = train_iterator.get_next()
            train_step(images, labels)
            train_writer.flush()

            if epoch == 0 and step == 0:
                model.summary()
                with val_writer.as_default():
                    tf.summary.scalar('loss',  train_loss.result(), step=offset)
                    tf.summary.scalar('accuracy', train_acc.result(), step=offset)
                    val_writer.flush()

            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=offset+step)
                tf.summary.scalar('accuracy', train_acc.result(), step=offset+step)
                if params.model_type == 'bnn':
                    tf.summary.scalar('kl_loss', kl_loss.result(), step=offset+step)
                    tf.summary.scalar('nll_loss', nll_loss.result(), step=offset+step)
                train_writer.flush()

            if (step+1) % 100 == 0:
                logging.info('Epoch: {}, Batch index: {}, '
                        'train loss: {:.3f}, train accuracy: {:.3%}'
                        .format(epoch, step + 1, 
                        train_loss.result(), 
                        train_acc.result()))
                if params.model_type == 'bnn':
                    logging.info('kl loss: {:.3f}, nll loss: {:.3f}'
                                 .format(kl_loss.result(),
                                         nll_loss.result()))

        corr_ls, total_ls = [[0 for m in params.brand_models] for i in range(2)]
        for step in trange(num_val_steps):
            images, labels = val_iterator.get_next()
            c, t = val_step(images, labels)
            corr_ls += c
            total_ls += t

        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=offset+num_train_steps)
            tf.summary.scalar('accuracy', val_acc.result(), step=offset+num_train_steps)
            val_writer.flush()

        logging.info('val loss: {:.3f}, validation accuracy: {:.3%}'.format(
                val_loss.result(), val_acc.result()))

        for m, c, t in zip(params.models, corr_ls, total_ls):
            logging.info('{} accuracy: {:.3%}'.format(m, c / t))
        logging.info('\n')

        # save the best model regarding to train acc
        ckpt.step.assign_add(1)
        
        # if val_acc.result() >= best_acc and \
        if val_loss.result() <= best_loss:
            save_path = manager.save()
            best_acc = val_acc.result()
            best_loss = val_loss.result()
            stop_count = 0
            logging.info("Saved checkpoint for epoch {}: {}\n".format(epoch, save_path))
        # early stopping after 10 epochs
        # elif epoch > 10:
        else:
            stop_count += 1
        
        if stop_count >= params.patience:
            break

    logging.info('\nFinished training\n')

if __name__ == '__main__':
    log = 'results/' + params.database + '/' + params.model_type
    tb_log = 'logs/' + params.database + '/' + params.model_type
    ckpt_dir = 'ckpts/' + params.database + '/' + params.model_type
    for path in [log, tb_log, ckpt_dir]:
        p_dir = os.path.dirname(path)
        if not os.path.exists(p_dir):
            os.makedirs(p_dir)
    build_and_train(log+'.log', tb_log, ckpt_dir)
