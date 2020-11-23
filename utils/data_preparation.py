import os
import numpy as np
import tensorflow as tf
import cv2
from functools import partial
from multiprocessing import Pool
from skimage import io, filters, img_as_ubyte, img_as_float64
from skimage.util import random_noise
from tqdm import tqdm, trange
AUTOTUNE = tf.data.experimental.AUTOTUNE


def parse_image(img_path, brand_models):
    """
    read label and covert to onehot vector, read images from
    paths, convert to the range 0-255.
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

def build_dataset(patch_dir, brand_models,
                dataset_id, batch_size, 
                img_paths=None, class_imbalance=False,
                degradation=None, factor=None):
    """
    build train, validation, test dataset as well as the dataset for different experiments.
    Args:
        patch_dir: the directory storing the extracted patches.
        brand_models: a list of the targeted camera models' name.
        dataset_id: the type of dataset.
        batch_size: desired batch size of the dataset.
        img_paths: image paths.
        class_imbalance: if true, use oversampling the monority class.
        degradation: the type of degradation, e.g. jpeg, blur and noise.
        factor: the parameter controls the degradation, quality factor for jpeg 
                and standard deviation of Gaussian for both blur and noise.
    Returns:
        iterator: the iterator of the generated dataset.
    """ 

    # create training set
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
            # if not class_imbalance, the dataset is then enforced to be even.
            dataset = (tf.data.Dataset.list_files(
                    os.path.join(patch_dir, 'train')+'/*/*')
                    .repeat()
                    # whole dataset into the buffer ensures good shuffling
                    .shuffle(buffer_size=1000) 
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
                .repeat()
                .map(partial(parse_image, brand_models=brand_models), 
                        num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE))
    # create dataset for degradation experiment.
    elif dataset_id == 'degradation':
        dataset = (tf.data.Dataset.list_files(
                os.path.join("data/degradation", '_'.join('dresden', degradation), factor)+'/*/*')
                .repeat()
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

def degradate(img_path_ls, img_root, database,
                degradation_id, factor):
    """
    degrade multiple images with post-processing operation.
    Args:
        img_path_ls: list of images paths for degradation.
        img_root: root directory for storing images.
        database: name of the database.
        degradation_id: type of degradation to perform.
        factor: paramter controled the degradation, for 
                jpeg, it is quality factor; for Gaussian 
                noise and Gaussian blur, it is the standard 
                deviation of the Gaussian distribution.
    Returns:
        target_path_ls
    """
    target_dir = os.path.join(img_root,
                    '_'.join([database, degradation_id]),
                    '{}'.format(factor))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    args_ls = []
    for img_path in img_path_ls:
        args_ls += [{'img_path': img_path,
                    'target_dir': target_dir,
                    'post_processing':degradation_id,
                    'factor': factor}]
    target_path_ls = []
    with Pool() as pool:
        target_path_ls.extend(pool.map(post_processing, args_ls))

    return target_path_ls


def post_processing(arg):
    """
    offline implementation for post processing. The offline version 
    images have different value compared to directly use the post-processed 
    float outputs of images (offline images are all integer values). Therefore, 
    offline is closer to real world case for camera model identification. 
    (analyze a post processed image instead of reading a image then adding post process.)
    Args:
        img_path: full paths of the source images.
        target_dir: root directory storing post-processed data.
        post_process: 'jpeg', covert to jpeg image from .png;
                      'blur', add gaussian blur;
                      'noise', add gaussian noise.
    Return:
        target_path: path of the saved post process images.
    """
    image_name = os.path.split(arg['img_path'])[-1]
    target_path = os.path.join(arg['target_dir'],
                                os.path.split(os.path.dirname(arg['img_path']))[-1],
                                image_name)
    out_fulldir = os.path.split(target_path)[0]
    if not os.path.exists(out_fulldir):
        os.makedirs(out_fulldir, exist_ok=True)
    # using jpeg compression
    if arg['post_processing'] == 'jpeg':
        target_path = os.path.splitext(target_path)[0] + '.jpg'
        if not os.path.exists(target_path):
            img = io.imread(arg['img_path'])
            io.imsave(target_path, img, plugins='pil', 
                        quality=arg['factor'], check_contrast=False)
    # adding Gaussian noise
    elif arg['post_processing'] == 'noise':
        if not os.path.exists(target_path):
            img = io.imread(arg['img_path'])
            noisy = random_noise(img, mode='gaussian', 
                                    mean=0, var=arg['factor']**2)
            io.imsave(target_path, img_as_ubyte(noisy), 
                        check_contrast=False)
    # adding Gaussian blur
    elif arg['post_processing'] == 'blur':
        if not os.path.exists(target_path): 
            img = io.imread(arg['img_path'])
            blur = cv2.GaussianBlur(img, (5,5), arg['factor'])
            # blur = filters.gaussian(img, sigma=arg['factor'], 
            #                         truncate=2.0)
            io.imsave(target_path, img_as_ubyte(blur),
                        check_contrast=False)
    # adding salt and peppers noise
    elif arg['post_processing'] == 's&p':
        if not os.path.exists(target_path):
            img = io.imread(arg['img_path'])
            noisy = random_noise(img, mode='s&p', 
                                    amount=arg['factor'])
            io.imsave(target_path, img_as_ubyte(noisy), 
                          check_contrast=False)
    return target_path

