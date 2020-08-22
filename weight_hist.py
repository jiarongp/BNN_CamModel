import os
import data_preparation as dp
import tensorflow as tf
import model_lib
import datetime
import params
from tqdm import trange
keras = tf.keras
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model = model_lib.bnn(100)
ckpt_dir = 'ckpts/' + 'dresden' + '/' + 'bnn'
# save model to a checkpoint
ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), 
        optimizer=keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
        net=model)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
train_writer = tf.summary.create_file_writer(train_log_dir)

image = tf.random.normal([1, 256, 256, 1])
logit = model(image)

with train_writer.as_default():
    for t_w in model.trainable_weights:
        if 'kernel' in t_w.name:
            tf.summary.histogram(t_w.name, t_w, step=1)
train_writer.flush()

print('... Done')