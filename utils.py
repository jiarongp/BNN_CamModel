import tensorflow as tf
import os
import urllib
import pandas as pd
import numpy as np
import params
import fnmatch
import proc
from tqdm import tqdm
from skimage import io
AUTOTUNE = tf.data.experimental.AUTOTUNE

def collect_dataset(data, images_dir):
    """Download data from the input csv to specific directory
    Args:
        data: a csv file storing the dataset with filename, model, brand and etc.
        images_dir: target root directory for the downloaded images
    Return:
        path_list: a list of paths of images. For example: 'image_dir/brand_model/filname.jpg'
    """
    csv_rows = []
    path_list = []
    brand_models = params.brand_models
    dirs = [os.path.join(params.dresden_images_dir, d) for d in brand_models]
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)

    for i in range((data.shape[0])): 
        csv_rows.append(list(data.iloc[i, :]))

    for csv_row in tqdm(csv_rows):
        filename, brand, model = csv_row[0:3]
        url = csv_row[-1]
        brand_model = '_'.join([brand, model])
        image_path = os.path.join(images_dir, brand_model, filename)
        if not os.path.exists(image_path):
            print('Downloading {:}'.format(filename))
            urllib.request.urlretrieve(url, image_path)
        path_list.append(image_path)

    print('Number of images: {:}\n'.format(len(path_list)))
    return path_list


def split_dataset(img_list, seed=42):
    """Split dataset into train, validation and test
    Args:
        img_list: the dataset need to be split, which is a list of paths of images
        seed: random seed for split.
    Return:
        split_ds: a list has shape [# of models, [# of train, # of val, # of test]], contains the 
                  full relative paths of images.
        weights: it's for class imbalance correction
    """
    num_test = int(len(img_list) * 0.15)
    num_val = num_test
    num_train = len(img_list) - num_test - num_val
    np.random.seed(seed)
    np.random.shuffle(img_list)
    train_list = img_list[0:num_train]
    val_list = img_list[num_train:(num_train + num_val)]
    test_list = img_list[(num_train + num_val):]
    # print out the split information
    split_ds = []
    weights = []
    model_list = params.brand_models
    for model in model_list:
        train = fnmatch.filter(train_list, '*' + model + '*')
        print("{} in training set: {}.".format(model, len(train)))
        val = fnmatch.filter(val_list, '*' + model + '*')
        print("{} in validation set: {}.".format(model, len(val)))
        test = fnmatch.filter(test_list, '*' + model + '*')
        print("{} in test set: {}.\n".format(model, len(test)))
        split_ds.append([train, val, test])
        #(1 / neg)*(total)/2.0 
        weights.append((1/len(train + val + test))*(len(img_list))/ 2.0)
    return split_ds, weights

def collect_split_extract():
    # collect data if not downloaded
    data = pd.read_csv(params.dresden)
    data = data[([m in params.models for m in data['model']])]
    image_paths = collect_dataset(data, params.dresden_images_dir)

    # split dataset in train, val and test
    split_ds, weights = split_dataset(image_paths)
    class_weight = {}
    for i in range(len(params.brand_models)):
        class_weight[i] = weights[i]
    # extract patches from full-sized images
    for i in range(len(params.brand_models)):
        print("... Extracting patches from {} images".format(params.brand_models[i]))
        proc.patch(path=split_ds[i][0], dataset='train')
        proc.patch(path=split_ds[i][1], dataset='val')
        proc.patch(path=split_ds[i][2], dataset='test')
        print("... Done\n")

    return class_weight


def string_to_onehot(input, vocab):
    matches = tf.stack([tf.equal(input, s) for s in vocab], axis=-1)
    onehot = tf.cast(matches, tf.float32)
    return onehot


def parse_image(img_path):
    label = tf.strings.split(img_path, os.path.sep)[-2]
    onehot = string_to_onehot(label, params.brand_models)
    # load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=1)
    # image covert to tf.float32 and /255.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, [params.IMG_HEIGHT, params.IMG_WIDTH])
    return img, onehot


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    # ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    # Applying the Dataset.repeat() transformation with no arguments will repeat the input indefinitely.
    # ds = ds.repeat()
    ds = ds.batch(params.BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

def build_dataset():
    # zip image and label
    # The Dataset.map(f) transformation produces a new dataset by applying a 
    # given function f to each element of the input dataset. 
    train_set = tf.data.Dataset.list_files(params.patches_dir + '/train/*/*')
    train_ds = train_set.map(parse_image, num_parallel_calls=AUTOTUNE)
    val_set = tf.data.Dataset.list_files(params.patches_dir + '/val/*/*')
    val_ds = val_set.map(parse_image, num_parallel_calls=AUTOTUNE)
    test_set = tf.data.Dataset.list_files(params.patches_dir + '/val/*/*')
    test_ds = test_set.map(parse_image, num_parallel_calls=AUTOTUNE)

    train_ds = prepare_for_training(train_ds)
    val_ds = prepare_for_training(val_ds)
    test_ds = prepare_for_training(test_ds)
    return train_ds, val_ds, test_ds