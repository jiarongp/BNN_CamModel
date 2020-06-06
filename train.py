import tensorflow as tf
import tensorflow_probability as tfp
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import params
import proc
import model
import utils
from tensorflow_probability import distributions as tfd

AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def main():
    data = pd.read_csv(params.dresden)
    data = data[([m in params.models for m in data['model']])]
    image_paths = utils.collect(data, params.dresden_images_dir)
    # split dataset in train, val and test
    split_ds, weights = utils.split(image_paths)
    # extract patches from full-sized images
    for i in range(len(params.brand_models)):
        print("... Extracting patches from {} images".format(params.brand_models[i]))
        proc.patch(path=split_ds[i][0], dataset='train')
        proc.patch(path=split_ds[i][1], dataset='val')
        proc.patch(path=split_ds[i][2], dataset='test')
        print("... Done\n")

    list_ds = tf.data.Dataset.list_files(params.patches_dir + '/train/*/*')
    train_set = list_ds.map(utils.process_path, num_parallel_calls=AUTOTUNE)
    list_ds = tf.data.Dataset.list_files(params.patches_dir + '/val/*/*')
    val_set = list_ds.map(utils.process_path, num_parallel_calls=AUTOTUNE)
    list_ds = tf.data.Dataset.list_files(params.patches_dir + '/val/*/*')
    test_set = list_ds.map(utils.process_path, num_parallel_calls=AUTOTUNE)

    train_set.cache()

    # train_seq = utils.ImageSequence(data=train_set, batch_size=params.BATCH_SIZE)
    # test_seq = utils.ImageSequence(data=test_set, batch_size=params.BATCH_SIZE)
    # bnn = model.create_model()
    # bnn.build(input_shape=[None, 28, 28, 1])


if __name__ == "__main__":
    main()