import tensorflow as tf
import os
import urllib
import pandas as pd
import numpy as np
import params
import fnmatch
from tqdm import tqdm
from skimage import io

def collect(data, images_dir):
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

    # count = 0
    for csv_row in tqdm(csv_rows):
        filename, brand, model = csv_row[0:3]
        url = csv_row[-1]
        brand_model = '_'.join([brand, model])
        image_path = os.path.join(images_dir, brand_model, filename)
        if not os.path.exists(image_path):
            print('Downloading {:}'.format(filename))
            urllib.request.urlretrieve(url, image_path)
        path_list.append(image_path)

        # try:
        #     if not os.path.exists(image_path):
        #         print('Downloading {:}'.format(filename))
        #         urllib.request.urlretrieve(url, image_path)
        #     # Load the image and check its dimensions
        #     img = io.imread(image_path)
        #     if img is None or not isinstance(img, np.ndarray):
        #         print('Unable to read image: {:}'.format(filename))
        #         # removes (deletes) the file path
        #         os.unlink(image_path)
        #     # if the size of all images are not zero, then append to the list
        #     if all(img.shape[:2]):
        #         count += 1
        #         path_list.append(image_path)
        #     else:
        #         print('Zero-sized image: {:}'.format(filename))
        #         os.unlink(image_path)

        # except IOError:
        #     print('Unable to decode: {:}'.format(filename))
        #     os.unlink(image_path)

        # except Exception as e:
        #     print('Error while loading: {:}'.format(filename))
        #     if os.path.exists(image_path):
        #         os.unlink(image_path)

    print('Number of images: {:}'.format(len(path_list)))
    return path_list

def split(img_list, seed=42):
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

def process_path(img_path):
    label = tf.strings.split(img_path, os.path.sep)[-2]
    # load the raw data from the file as a string
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=1)
    # img = tf.image.resize(img, [params.IMG_HEIGHT, params.IMG_WIDTH])
    return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(params.BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        # images, labels = data
        self.images, self.labels = ImageSequence.__preprocessing(data)
        self.batch_size = batch_size

    @staticmethod
    def __preprocessing(data):
        """Preprocessing images and labels data
        
        Returns:
            images: image data, normalized and expanded for convolutional
                    neural network input.
            labels: labels data (0-9) as one-hot(categorical) values.
        """
        images, labels = data
        images = images / 255.
        # images = images[..., tf.newaxis]
        labels = tf.keras.utils.to_categorical(labels)
        return images, labels

    def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return batch_x, batch_y