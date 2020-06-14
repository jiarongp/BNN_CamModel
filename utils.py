import tensorflow as tf
import os
import logging
import urllib
import pandas as pd
import numpy as np
import params
import fnmatch
import proc
import time
from tqdm import tqdm
from skimage import io
AUTOTUNE = tf.data.experimental.AUTOTUNE


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


def _collect_dataset(data, images_dir):
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
            logging.info('Downloading {:}'.format(filename))
            urllib.request.urlretrieve(url, image_path)
        path_list.append(image_path)
        time.sleep(0.01)

    logging.info('Number of images: {:}\n'.format(len(path_list)))
    return path_list


def _split_dataset(img_list, seed=42):
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
        logging.info("{} in training set: {}.".format(model, len(train)))
        val = fnmatch.filter(val_list, '*' + model + '*')
        logging.info("{} in validation set: {}.".format(model, len(val)))
        test = fnmatch.filter(test_list, '*' + model + '*')
        logging.info("{} in test set: {}.\n".format(model, len(test)))
        split_ds.append([train, val, test])
        #(1 / neg)*(total)/2.0 
        weights.append((1/len(train + val + test))*(len(img_list))/ 2.0)
    return split_ds, weights

def collect_split_extract():
    # collect data if not downloaded
    data = pd.read_csv(params.dresden)
    data = data[([m in params.models for m in data['model']])]
    image_paths = _collect_dataset(data, params.dresden_images_dir)

    # split dataset in train, val and test
    split_ds, weights = _split_dataset(image_paths)
    class_weight = {}
    for i in range(len(params.brand_models)):
        class_weight[i] = weights[i]
    # extract patches from full-sized images
    for i in range(len(params.brand_models)):
        logging.info("... Extracting patches from {} images".format(params.brand_models[i]))
        proc.patch(path=split_ds[i][0], dataset='train')
        proc.patch(path=split_ds[i][1], dataset='val')
        proc.patch(path=split_ds[i][2], dataset='test')
        logging.info("... Done\n")

    return class_weight


def _parse_image(img_path):
    label = tf.strings.split(img_path, os.path.sep)[-2]

    matches = tf.stack([tf.equal(label, s) for s in params.brand_models], axis=-1)
    onehot_label = tf.cast(matches, tf.float32)

    # load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=1)
    # image covert to tf.float32 and /255.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, [params.IMG_HEIGHT, params.IMG_WIDTH])
    return img, onehot_label


def _train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def build_dataset(dataset_name):
    # zip image and label
    # The Dataset.map(f) transformation produces a new dataset by applying a 
    # given function f to each element of the input dataset. 
    if dataset_name == 'train':
        dataset = (tf.data.Dataset.list_files(params.patches_dir + '/train/*/*')
            .repeat()
            .shuffle(buffer_size=1000)  # whole dataset into the buffer ensures good shuffling
            .map(_parse_image, num_parallel_calls=AUTOTUNE)
            .batch(params.BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)  # make sure you always have one batch ready to serve
        )
    elif dataset_name == 'val':
        dataset = (tf.data.Dataset.list_files(params.patches_dir + '/val/*/*')
            .repeat()
            .map(_parse_image, num_parallel_calls=AUTOTUNE)
            .batch(params.BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.list_files(params.patches_dir + '/test/*/*')
            .map(_parse_image, num_parallel_calls=AUTOTUNE)
            .batch(params.BATCH_SIZE)
            .prefetch(buffer_size=AUTOTUNE)  # make sure you always have one batch ready to serve
        )

    iterator = iter(dataset)
    return iterator