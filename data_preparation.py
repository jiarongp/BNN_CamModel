import os
import fnmatch
import numpy as np
import pandas as pd
import logging
import params
from tqdm import tqdm 
import urllib
import tensorflow as tf
# import time
from multiprocessing import Pool
from skimage.util.shape import view_as_blocks
from skimage import io

AUTOTUNE = tf.data.experimental.AUTOTUNE

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
        # time.sleep(0.01)

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


def _patchify(img_path, patch_span=params.patch_span):
    """Separate the full-sized image into 256 x 256 image patches. By default, the full-sized
    images is split into 25 patches.
    Args:
        img_path: the path of the source image.
        patch_span: decide the number of patches, which is patch_span^2.
    Return:
        patches: 25 patches
    """
    img = io.imread(img_path)
    if img is None or not isinstance(img, np.ndarray):
        print('Unable to read the image: {:}'.format(img_path))

    center = np.divide(img.shape[:2], 2).astype(int)
    start = np.subtract(center, patch_span/2).astype(int)
    end = np.add(center, patch_span/2).astype(int)
    sub_img = img[start[0]:end[0], start[1]:end[1]]
    sub_img = np.asarray(sub_img)
    patches = view_as_blocks(sub_img[:, :, 1], (256, 256))
    return patches


def _patch(path, dataset):
    """call the extract function to extract patches from full-sized image
    Args:
        path: paths for images needed to be split into patches
        dataset: one of ['train', 'val', 'test']
    """
    imgs_list = []
    for img_path in path:
        imgs_list += [{'dataset':dataset,
                    'img_path':img_path
                    }]
    num_processes = 8
    pool = Pool(processes=num_processes)
    pool.map(_extract, imgs_list)


def _extract(args):
    """extract patches from full-sized image
    Args:
        dataset: dataset the image belongs to, 'train', 'val' or 'test'
        img_path: full paths of the source images
    Return:
        output_rel_paths: the paths of extracted patches. For example:
                          'train/brand_model/filename_idx.png'
    """
    # 'train/Agfa_DC-504/Agfa_DC-504_0_1_00.png' for example,
    # last part is the patch idex.
    # Use PNG for losslessly storing images
    output_rel_paths = [os.path.join(args['dataset'],
                        os.path.split(os.path.dirname(args['img_path']))[-1],
                        os.path.splitext(os.path.split(args['img_path'])[-1])[0]+'_'+'{:02}'.format(patch_idx) + '.png')\
                        for patch_idx in range(params.patch_num)]
    read_img = False
    for out_path in output_rel_paths:
        out_fullpath = os.path.join(params.patches_dir, out_path)

        # if there is no this path, then we have to read images
        if not os.path.exists(out_fullpath):
            read_img = True
            break
    if read_img:
        patches = _patchify(args['img_path']).reshape((-1, 256, 256))
    
        for out_path, patch in zip(output_rel_paths, patches):
            out_fullpath = os.path.join(params.patches_dir, out_path)
            # the diretory of the patches images
            out_fulldir = os.path.split(out_fullpath)[0]
            if not os.path.exists(out_fulldir):
                os.makedirs(out_fulldir)
            if not os.path.exists(out_fullpath):
                io.imsave(out_fullpath, patch, check_contrast=False)
    return output_rel_paths


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
        _patch(path=split_ds[i][0], dataset='train')
        _patch(path=split_ds[i][1], dataset='val')
        _patch(path=split_ds[i][2], dataset='test')
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