import os
import fnmatch
import numpy as np
import pandas as pd
import logging
import params
import urllib
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm, trange
# from multiprocessing import Pool
from skimage.util.shape import view_as_blocks
from skimage import io
AUTOTUNE = tf.data.experimental.AUTOTUNE

def collect_dataset(data, images_dir, brand_models, download=False):
    """Download data from the input csv to specific directory
    Args:
        data: a csv file storing the dataset with filename, model, brand and etc.
        images_dir: target root directory for the downloaded images.
        brand_models: the brand_model name of the target images.
        download: if your are sure the data are already there, just use False.
    Return:
        path_list: a list of paths of images. For example: 'image_dir/brand_model/filname.jpg'
    """
    csv_rows = []
    path_list = []
    dirs = [os.path.join(images_dir, d) for d in brand_models]
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)
    data = (data[['filename', 'brand', 'model', 'url']] 
            if params.database == 'dresden' 
            else data[['File', 'TIFF', 'Device']])
    for i in range((data.shape[0])): 
        csv_rows.append(list(data.iloc[i, :]))

    for csv_row in tqdm(csv_rows, position=0, leave=True):
        if params.database == 'dresden':
            filename, brand, model = csv_row[0:3]
            url = csv_row[-1]
            brand_model = '_'.join([brand, model])
        else:
            filename, url, brand_model = csv_row
            brand_model = '_'.join(brand_model.split(' '))
            filename = '{}_{}.{}'.format(brand_model, filename, 'TIF')
        image_path = os.path.join(images_dir, brand_model, filename)

        if download:
            try:
                if not os.path.exists(image_path):
                    tqdm.write('Downloading {:}'.format(filename))
                    urllib.request.urlretrieve(url, image_path)
                # Load the image and check its dimensions
                img = io.imread(image_path)
                if img is None or not isinstance(img, np.ndarray):
                    logging.info('Unable to read image: {:}'.format(filename))
                    # removes (deletes) the file path
                    os.unlink(image_path)
                # if the size of all images are not zero, then append to the list
                if all(img.shape[:2]):
                    path_list.append(image_path)
                else:
                    logging.info('Zero-sized image: {:}'.format(filename))
                    os.unlink(image_path)

            except IOError:
                logging.info('Unable to decode: {:}'.format(filename))
                os.unlink(image_path)

            except Exception as e:
                logging.info('Error while loading: {:}'.format(filename))
                if os.path.exists(image_path):
                    os.unlink(image_path)
        else:
            path_list.append(image_path)
            
    logging.info('Number of images: {:}\n'.format(len(path_list)))
    return path_list


def split_dataset(img_list, brand_models, seed=42):
    """Split dataset into train, validation and test
    Args:
        img_list: the dataset need to be split, which is a list of paths of images
        seed: random seed for split.
    Return:
        split_ds: a list has shape [# of models, [# of train, # of val, # of test]], contains the 
                  full relative paths of images.
    """
    split_ds = []
    if params.even_database:
        np.random.seed(42)
        model_images, num_images = [], []
        for model in params.brand_models:
            images = fnmatch.filter(img_list, '*' + model + '*')
            model_images.append(images)
            num_images.append(len(images))
            
        num_ds = min(num_images)
        num_val = int(0.1 * num_ds)
        num_train = num_ds - 2 * num_val

        for model, images in zip(params.brand_models, model_images):
            np.random.shuffle(images)
            train = images[0:num_train]
            logging.info("{} in training set: {}.".format(model, len(train)))
            val = images[num_train:(num_train + num_val)]
            logging.info("{} in validation set: {}.".format(model, len(val)))
            test = images[(num_train + num_val):(num_train + 2*num_val)]
            logging.info("{} in test set: {}.\n".format(model, len(test)))
            split_ds.append([train, val, test])
    else:
        # num_test equals to num_val
        num_test = num_val = int(len(img_list) * 0.15)
        num_train = len(img_list) - num_test - num_val

        np.random.seed(seed)
        np.random.shuffle(img_list)

        train_list = img_list[0:num_train]
        val_list = img_list[num_train:(num_train + num_val)]
        test_list = img_list[(num_train + num_val):]
        # print out the split information
        for model in brand_models:
            train = fnmatch.filter(train_list, '*' + model + '*')
            logging.info("{} in training set: {}.".format(model, len(train)))
            val = fnmatch.filter(val_list, '*' + model + '*')
            logging.info("{} in validation set: {}.".format(model, len(val)))
            test = fnmatch.filter(test_list, '*' + model + '*')
            logging.info("{} in test set: {}.\n".format(model, len(test)))
            split_ds.append([train, val, test])
    return split_ds


def patchify(img_path, patch_span=params.patch_span):
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


def patch(path, data_id, parent_dir):
    """call the extract function to extract patches from full-sized image
    Args:
        path: paths for images needed to be split into patches
        data_id: one of ['train', 'val', 'test']
    """
    imgs_list = []
    for img_path in path:
        imgs_list += [{'data_id':data_id,
                       'img_path':img_path,
                       'parent_dir':parent_dir}]
    # num_processes = 4
    # pool = Pool(processes=num_processes)
    # pool.map(extract, imgs_list)
    for img in imgs_list:
        extract(img)


def extract(args):
    """extract patches from full-sized image
    Args:
        data_id: dataset the image belongs to, 'train', 'val' or 'test'
        img_path: full paths of the source images
    Return:
        output_rel_paths: the paths of extracted patches. For example:
                          'train/brand_model/filename_idx.png'
    """
    # 'train/Agfa_DC-504/Agfa_DC-504_0_1_00.png' for example,
    # last part is the patch idex.
    # Use PNG for losslessly storing images

    output_rel_paths = [os.path.join(args['data_id'],
                        os.path.split(os.path.dirname(args['img_path']))[-1],
                        os.path.splitext(os.path.split(args['img_path'])[-1])[0]+'_'+'{:02}'
                        .format(patch_idx)
                        + ('.png' if params.database == 'dresden' 
                           else '.TIF'))
                        for patch_idx in range(params.patch_num)]
    
    read_img = False
    parent_dir = args['parent_dir']
    for out_path in output_rel_paths:
        out_fullpath = os.path.join(parent_dir, out_path)

        # if there is no this path, then we have to read images
        if not os.path.exists(out_fullpath):
            read_img = True
            break
    if read_img:
        patches = patchify(args['img_path']).reshape((-1, 256, 256))
        for out_path, patch in zip(output_rel_paths, patches):
            out_fullpath = os.path.join(parent_dir, out_path)
            # the diretory of the patches images
            out_fulldir = os.path.split(out_fullpath)[0]
            if not os.path.exists(out_fulldir):
                os.makedirs(out_fulldir)
            if not os.path.exists(out_fullpath):
                io.imsave(out_fullpath, patch, check_contrast=False)
    # return output_rel_paths


def collect_split_extract(parent_dir, download_images=False):
    # collect data if not downloaded
    data = pd.read_csv(params.ds_csv)
    if params.database=='dresden':
        data = data[([m in params.models for m in data['model']])]
    image_paths = collect_dataset(data, 
                                  params.ds_image_dir,
                                  params.brand_models,
                                  download=download_images)

    # split dataset in train, val and test
    split_ds = split_dataset(image_paths, 
                             brand_models=params.brand_models)
    # extract patches from full-sized images
    for i in range(len(params.brand_models)):
        logging.info("... Extracting patches from {} images".format(params.brand_models[i]))
        patch(path=split_ds[i][0], data_id='train', parent_dir=parent_dir)
        patch(path=split_ds[i][1], data_id='val', parent_dir=parent_dir)
        patch(path=split_ds[i][2], data_id='test', parent_dir=parent_dir)
        logging.info("... Done\n")

def collect_unseen(download):
    # collect odd data
    data = pd.read_csv(params.ds_csv)
    if params.database == 'dresden':
        data = data[([m in params.unseen_models for m in data['model']])]
    elif params.database == 'RAISE':
        device = [' '.join([b, m]) for (b, m) in zip(params.unseen_brands, params.unseen_models)]
        data = data[([d in device for d in data['Device']])]

    image_paths = collect_dataset(data, 
                                 params.ds_image_dir,
                                 params.unseen_brand_models,
                                 download=download)

    patch(path=image_paths, data_id='test', parent_dir=params.unseen_dir)
    print("... Done\n")

def parse_image(img_path, post_processing=None):
    label = tf.strings.split(img_path, os.path.sep)[-2]
    matches = tf.stack([tf.equal(label, s) for s in params.brand_models], axis=-1)
    onehot_label = tf.cast(matches, tf.float32)

    # load the raw data from the file as a string
    if params.database == 'RAISE':
        # image = [io.imread(path.numpy().decode('utf8')) for path in img_path]
        # image = tf.stack(image, axis=0)
        image = io.imread(img_path.numpy().decode('utf8'))
        image = image[..., None]
    else:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image)

    # image covert to tf.float32 and /255.
    image = tf.image.convert_image_dtype(image, tf.float32)

    if post_processing == 'jpeg':
        image = tf.image.adjust_jpeg_quality(image, 70)
    elif post_processing == 'blur':
        image = tfa.image.gaussian_filter2d(image, 
                                            filter_shape=[5, 5],
                                            sigma=1.1)
    elif post_processing == 'noise':
        # image must be scaled in [0, 1]
        noise = tf.random.normal(shape=tf.shape(image), 
                                 mean=0.0, stddev=(2)/(255),
                                 dtype=tf.float32)
        image += noise
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    # img = tf.image.resize(img, [params.IMG_HEIGHT, params.IMG_WIDTH])
    return image, onehot_label


def build_dataset(data_id, class_imbalance=False):
    # zip image and label
    # The Dataset.map(f) transformation produces a new dataset by applying a 
    # given function f to each element of the input dataset.
    fn = lambda x: tf.py_function(parse_image, inp=[x], Tout=[tf.float32, tf.float32])
    if data_id == 'train':
        # use oversampling to counteract the class imbalance
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#oversampling
        if class_imbalance:
            class_datasets = []
            for m in params.brand_models:
                class_dataset = (tf.data.Dataset.list_files(params.patch_dir + '/train/' + m + '/*') \
                                 .shuffle(buffer_size=1000).repeat())
                class_datasets.append(class_dataset)
            # uniformly samples in the class_datasets
            dataset = (tf.data.experimental.sample_from_datasets(class_datasets)
                       .map(parse_image if params.database == 'dresden'
                            else fn, num_parallel_calls=AUTOTUNE 
                            if params.database == 'dresden' else None)
                       .batch(params.BATCH_SIZE)
                       .prefetch(buffer_size=AUTOTUNE))  # make sure you always have one batch ready to serve
            
        else:
            dataset = (tf.data.Dataset.list_files(params.patch_dir + '/train/*/*')
                       .repeat()
                       .shuffle(buffer_size=1000)  # whole dataset into the buffer ensures good shuffling
                       .map(parse_image if params.database == 'dresden'
                            else fn, num_parallel_calls=AUTOTUNE 
                            if params.database == 'dresden' else None)
                       .batch(params.BATCH_SIZE)
                       .prefetch(buffer_size=AUTOTUNE))

    elif data_id == 'val':
        dataset = (tf.data.Dataset.list_files(params.patch_dir + '/val/*/*')
                   .repeat()
                   .map(parse_image if params.database == 'dresden'
                        else fn, num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)
                   .batch(params.BATCH_SIZE)
                   .prefetch(buffer_size=AUTOTUNE))

    elif data_id == 'test':
        dataset = (tf.data.Dataset.list_files(params.patch_dir + '/test/*/*')
                   .map(parse_image if params.database == 'dresden'
                        else fn, num_parallel_calls=AUTOTUNE 
                        if params.database == 'dresden' else None)
                   .batch(params.BATCH_SIZE)
                   .prefetch(buffer_size=AUTOTUNE))

    iterator = iter(dataset)
    return iterator


def split_image(filename, post_processing=None, show_image=True):
    label =  tf.strings.split(filename, os.sep)[-1]
    image = io.imread(filename)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if post_processing == 'jpeg':
        image = tf.image.adjust_jpeg_quality(image, 70)
    elif post_processing == 'blur':
        image = tfa.image.gaussian_filter2d(image, 
                                            filter_shape=[5, 5],
                                            sigma=1.1)
    elif post_processing == 'noise':
        image = add_gaussian_noise(image)
    elif post_processing == 'salt':
        image = add_salt_pepper_noise(image.numpy())
        
    image = tf.clip_by_value(image, 0.0, 1.0)
        
    if show_image:
        plt.figure()
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlabel(label.numpy().decode('utf-8'))

    # divide into patches
    # adaptive patchify
    v_patch_span = image.shape[0] // 256 * 256
    h_patch_span = image.shape[1] // 256 * 256
    patch_span = min([h_patch_span, v_patch_span, params.patch_span])
    
    center = np.divide(image.shape[:2], 2).astype(int)
    start = np.subtract(center, patch_span/2).astype(int)
    end = np.add(center, patch_span/2).astype(int)
    sub_img = image[start[0]:end[0], start[1]:end[1]]
    sub_img = np.asarray(sub_img)
    patches = view_as_blocks(sub_img[:, :, 1], (256, 256))

    images = patches.reshape((-1, 256, 256))
    images = images[..., tf.newaxis]
    return images, label