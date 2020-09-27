import os
import fnmatch
import numpy as np
import pandas as pd
import urllib
import tensorflow as tf
from skimage import io
from tqdm import tqdm, trange
from utils.log import write_log
from utils.patch import extract_patch
AUTOTUNE = tf.data.experimental.AUTOTUNE


class BaseDataLoader(object):
    def __init__(self, params):
        self.img_path_ls = []
        self.params = params
        self.brand_models = self.params.dataloader.brand_models
        self.log_file = self.params.log.log_file

    def split_dataset(self, seed):
        """Split dataset into train, validation and test.
        Args:
            seed: random seed for split.
        Return:
            split_ds: a list has shape [# of models, [# of train, # of val, # of test]], 
                    contains the full relative paths of images.
        """
        if not self.img_path_ls:
            raise Exception("!!! The list of image paths is empty")
        # num_test equals to num_val
        num_test = num_val = int(len(self.img_path_ls) * 0.1)
        num_train = len(self.img_path_ls) - num_test - num_val

        np.random.seed(seed)
        np.random.shuffle(self.img_path_ls)

        train_list = self.img_path_ls[0:num_train]
        val_list = self.img_path_ls[num_train:(num_train + num_val)]
        test_list = self.img_path_ls[(num_train + num_val):]
        # print out the split information
        for model in self.brand_models:
            train = fnmatch.filter(train_list, '*' + model + '*')
            msg = "{} in training set: {}.\n".format(model, len(train))
            write_log(self.log_file, msg)
            val = fnmatch.filter(val_list, '*' + model + '*')
            msg = "{} in validation set: {}.\n".format(model, len(val))
            write_log(self.log_file, msg)
            test = fnmatch.filter(test_list, '*' + model + '*')
            msg = "{} in test set: {}.\n\n".format(model, len(test))
            write_log(self.log_file, msg)
            self.split_ds.append([train, val, test])


class DresdenDataLoader(BaseDataLoader):
    def __init__(self, params):
        super(DresdenDataLoader, self).__init__(params)
        self.split_ds = []
        self.models = self.params.dataloader.models
        self.images_dir = self.params.dataloader.database_image_dir
        self.num_cls = len(self.brand_models)

    def collect_dataset(self):
        """Download data from the input csv to specific directory.
        Args:
            data: a csv file storing the dataset with filename, 
                    model, brand and etc.
            images_dir: target root directory for the downloaded images.
            brand_models: the brand_model name of the target images.
        Return:
            path_list: a list of paths of images. 
            For example: 'image_dir/brand_model/filname.jpg'.
        """
        csv_rows = []
        dirs = [os.path.join(self.images_dir, d) 
                for d in self.brand_models]
        download = False
        if not os.path.exists(self.images_dir):
            download = True
        for path in dirs:
            if not os.path.exists(path):
                os.makedirs(path)

        # collect data if not downloaded
        data = pd.read_csv(self.params.dataloader.database_csv)
        data = data[([m in self.models 
                    for m in data['model']])]
        data = data[['filename', 'brand', 'model', 'url']]
        for i in range((data.shape[0])): 
            csv_rows.append(list(data.iloc[i, :]))

        for csv_row in tqdm(csv_rows):
            filename, brand, model = csv_row[0:3]
            url = csv_row[-1]
            brand_model = '_'.join([brand, model])
            img_path = os.path.join(
                self.images_dir,
                brand_model, 
                filename)
            if download:
                try:
                    if not os.path.exists(img_path):
                        tqdm.write('Downloading {:}'.format(filename))
                        urllib.request.urlretrieve(url, img_path)
                    # Load the image and check its dimensions
                    img = io.imread(img_path)
                    if img is None or not isinstance(img, np.ndarray):
                        print('Unable to read image: {:}'.format(filename))
                        # removes (deletes) the file path
                        os.unlink(img_path)
                    # if the size of all images are not zero, then append to the list
                    if all(img.shape[:2]):
                        self.img_path_ls.append(img_path)
                    else:
                        print('Zero-sized image: {:}'.format(filename))
                        os.unlink(img_path)
                except IOError:
                    print('Unable to decode: {:}'.format(filename))
                    os.unlink(img_path)
                except Exception as e:
                    print('Error while loading: {:}'.format(filename))
                    if os.path.exists(img_path):
                        os.unlink(img_path)
            else:
                self.img_path_ls.append(img_path)
        msg = 'Number of images: {:}\n'.format(len(self.img_path_ls))
        write_log(self.log_file, msg)


    def load_data(self):
        # download images
        self.collect_dataset()
        # split into train, val and test 
        self.split_dataset(self.params.dataloader.random_seed)
        for i in range(self.num_cls):
            print("... Extracting patches from {} images\n"
                    .format(self.brand_models[i]))
            extract_patch(
                img_path_ls=self.split_ds[i][0], 
                ds_id='train', 
                patch_dir=self.params.dataloader.patch_dir,
                num_patch=self.params.dataloader.num_patch,
                extract_span=self.params.dataloader.extract_span)
            extract_patch(
                img_path_ls=self.split_ds[i][1], 
                ds_id='val', 
                patch_dir=self.params.dataloader.patch_dir,
                num_patch=self.params.dataloader.num_patch,
                extract_span=self.params.dataloader.extract_span)
            extract_patch(
                img_path_ls=self.split_ds[i][2], 
                ds_id='test', 
                patch_dir=self.params.dataloader.patch_dir,
                num_patch=self.params.dataloader.num_patch,
                extract_span=self.params.dataloader.extract_span)
            print("... Done\n")


class UnseenDresdenDataLoader(DresdenDataLoader):
    def __init__(self, params):
        super(UnseenDresdenDataLoader, self).__init__(params)
        self.brand_models = self.params.unseen_dataloader.brand_models
        self.images_dir = self.params.unseen_dataloader.database_image_dir
        self.patch_dir = self.params.unseen_dataloader.patch_dir
        self.num_patch = self.params.unseen_dataloader.num_patch
        self.extract_span = self.params.unseen_dataloader.extract_span

    def load_data(self):
        # download images
        self.collect_dataset()
        for brand_model in self.brand_models:
            img_names = os.listdir(os.path.join(
                            self.images_dir,
                            brand_model))
            img_path_ls = [os.path.join(self.images_dir, brand_model, name) 
                            for name in img_names]
            print("... Extracting patches from {} images\n"
                    .format(brand_model))
            extract_patch(
                img_path_ls=img_path_ls,
                ds_id='.', 
                patch_dir= self.patch_dir,
                num_patch=self.num_patch,
                extract_span=self.extract_span)
            print("... Done\n")

class KaggleDataLoader(UnseenDresdenDataLoader):
    def __init__(self, params):
        super(KaggleDataLoader, self).__init__(params)
        self.brand_models = self.params.kaggle_dataloader.brand_models
        self.images_dir = self.params.kaggle_dataloader.database_image_dir
        self.patch_dir = self.params.kaggle_dataloader.patch_dir
        self.num_patch = self.params.kaggle_dataloader.num_patch
        self.extract_span = self.params.kaggle_dataloader.extract_span