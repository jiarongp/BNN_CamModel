import model_lib
import os
import params
import data_preparation as dp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import utils
from tqdm import trange
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# import BNN model
train_size = 0
for m in params.brand_models:
    train_size += len(os.listdir(os.path.join(params.patches_dir, 'train', m)))
num_test_steps = (train_size + params.BATCH_SIZE - 1) // params.BATCH_SIZE 

model = model_lib.BNN(train_size)
ckpt = tf.train.Checkpoint(
    step=tf.Variable(1), 
    optimizer=tf.keras.optimizers.Adam(lr=params.HParams['init_learning_rate']), 
    net=model)
manager = tf.train.CheckpointManager(ckpt, './ckpts/BNN_num_examples_2', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

test_iterator = dp.build_dataset('test')

mc_s_prob = []
for i in trange(params.num_monte_carlo):
    softmax_prob_all, softmax_prob_right, softmax_prob_wrong, accuracy = [], [], [], []
    for step in range(num_test_steps):
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
    
    mc_s_prob.append(softmax_all) 

    accuracy = np.mean(accuracy)
    err = 100 - accuracy

mc_s_prob = np.asarray(mc_s_prob)
std_all = np.std(mc_s_prob, axis=1)