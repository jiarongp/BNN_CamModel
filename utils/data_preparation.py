import os
import numpy as np
import tensorflow as tf
from functools import partial
from skimage import io, filters, img_as_ubyte
from skimage.util import random_noise
from tqdm import tqdm, trange
AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_image(img_path, brand_models):
    """read label and covert to onehot vector, read images from
    paths, adding post-process to images(optional), convert to
    to the range 0-255.
    Args:
        img_path: full paths of the source images.
        post_process: 'jpeg', covert to jpeg image from .png;
                      'blur', add gaussian blur;
                      'noise', add gaussian noise.
    Return:
        image: decoded images.
        onehot_label: onehot label.
    """ 
    label = tf.strings.split(img_path, os.path.sep)[-2]
    matches = tf.stack([tf.equal(label, s) 
                        for s in brand_models], 
                        axis=-1)
    onehot_label = tf.cast(matches, tf.float32)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image)
    # image covert to tf.float32 and /255.
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)
    # image = tf.image.resize(image, [params.IMG_HEIGHT, params.IMG_WIDTH])
    return image, onehot_label

def build_dataset(patch_dir, 
                brand_models,
                dataset_id, 
                batch_size, 
                img_paths=None, 
                class_imbalance=False):
        if dataset_id == 'train':
            # use oversampling to counteract the class imbalance
            # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#oversampling
            if class_imbalance:
                class_datasets = []
                for m in brand_models:
                    class_dataset = (tf.data.Dataset.list_files(
                        os.path.join(patch_dir, 'train', m)+'/*')
                        .shuffle(buffer_size=1000).repeat())
                    class_datasets.append(class_dataset)
                # uniformly samples in the class_datasets
                dataset = (tf.data.experimental.sample_from_datasets(class_datasets)
                        .map(partial(parse_image, brand_models=brand_models), 
                                num_parallel_calls=AUTOTUNE)
                        .batch(batch_size)
                        .prefetch(buffer_size=AUTOTUNE))  
                        # make sure you always have one batch ready to serve
            else:
                dataset = (tf.data.Dataset.list_files(
                        os.path.join(patch_dir, 'train')+'/*/*')
                        .repeat()
                        .shuffle(buffer_size=1000)  
                        # whole dataset into the buffer ensures good shuffling
                        .map(partial(parse_image, brand_models=brand_models), 
                                num_parallel_calls=AUTOTUNE)
                        .batch(batch_size)
                        .prefetch(buffer_size=AUTOTUNE))
        elif dataset_id == 'val':
            dataset = (tf.data.Dataset.list_files(
                    os.path.join(patch_dir, 'val')+'/*/*')
                    .repeat()
                    .map(partial(parse_image, brand_models=brand_models), 
                            num_parallel_calls=AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))
        elif dataset_id == 'test':
            dataset = (tf.data.Dataset.list_files(
                    os.path.join(patch_dir, 'test')+'/*/*')
                    .map(partial(parse_image, brand_models=brand_models), 
                            num_parallel_calls=AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE))
        else:
            dataset = (tf.data.Dataset.from_tensor_slices(img_paths)
                        .repeat()                    
                        .map(partial(parse_image, brand_models=brand_models), 
                                num_parallel_calls=AUTOTUNE)
                        .batch(batch_size)
                        .prefetch(buffer_size=AUTOTUNE))
        iterator = iter(dataset)
        return iterator

def post_processing(img_paths, img_root, database,
                    post_processing, *args):
    """offline implementation for post processing. The offline version 
    images have different value compared to the online version (offline
    images are all integer values).offline is closer to real world case 
    for camera model identification. (analyze a post processed image not 
    read a image then adding post process.)
    Args:
        img_path: full paths of the source images.
        post_process: 'jpeg', covert to jpeg image from .png;
                      'blur', add gaussian blur;
                      'noise', add gaussian noise.
    Return:
        target_path: path of the saved post process images.
    """
    image_name = [os.path.split(path)[-1] for path in img_paths]
    target_dir = os.path.join(img_root,
                        '_'.join([database, post_processing]),
                        '{}'.format(args[0]))
    target_paths = [os.path.join(target_dir, 
                    name) for name in image_name]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if post_processing == 'jpeg':
        quality = args[0]
        target_paths = [os.path.splitext(path)[0] + '.jpg'
                       for path in target_paths]
        for img_path, j_path in zip(img_paths, target_paths):
            if not os.path.exists(j_path):
                im = io.imread(img_path)
                io.imsave(j_path, im, plugins='pil', 
                          quality=quality, check_contrast=False)
    # Adding Gaussian Noise
    elif post_processing == 'noise':
        mean = 0
        sigma = args[0]
        for img_path, n_path in zip(img_paths, target_paths):
            if not os.path.exists(n_path):
                image = io.imread(img_path)
                noisy = random_noise(image, mode='gaussian', 
                                     mean=0, var=sigma**2)
                io.imsave(n_path, img_as_ubyte(noisy), 
                          check_contrast=False)
    # Adding Gaussian Blur
    elif post_processing == 'blur':
        sigma = args[0]
        for img_path, b_path in zip(img_paths, target_paths):
            if not os.path.exists(b_path): 
                image = io.imread(img_path)
                blur = filters.gaussian(image, sigma=sigma, 
                                        truncate=2.0)
                io.imsave(b_path, img_as_ubyte(blur), 
                          check_contrast=False)
    elif post_processing == 's&p':
        mean = 0
        amount = args[0]
        for img_path, s_path in zip(img_paths, target_paths):
            if not os.path.exists(s_path):
                image = io.imread(img_path)
                noisy = random_noise(image, mode='s&p', 
                                     amount=amount)
                io.imsave(s_path, img_as_ubyte(noisy), 
                          check_contrast=False)
    return target_paths


