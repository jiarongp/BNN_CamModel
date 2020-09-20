import os
import numpy as np
from utils.data_preparation import build_dataset, post_processing


class Experiment(object):
    def __init__(self, params):
        self.params = params
        self.random_seed = self.params.experiment.random_seed
        self.in_img_paths, self.num_test_batches = \
                self.aligned_dataset(os.path.join(
                    self.params.dataloader.patch_dir, 'test'),
                    self.params.dataloader.brand_models,
                    seed=self.random_seed)
        self.in_iter = build_dataset(
                        self.params.dataloader.patch_dir,
                        self.params.dataloader.brand_models,
                        "in distribution",
                        self.in_img_paths)

    def aligned_dataset(self,
            patch_dir, brand_models, batch_size,
            num_batches=None, seed=42):
        """
        set a fix subset of total test dataset, so that:
        1. each class has same size of test images
        2. each monte carlo draw has the same images as input
        3. random seed controls the how to sample this subset
        """
        # default for 'test' data
        np.random.seed(seed)
        image_paths, num_images, img_paths = [], [], []
        for model in brand_models:
            images = os.listdir(os.path.join(patch_dir, model))
            paths = [os.path.join(patch_dir, model, img) for img in images]
            image_paths.append(paths)
            num_images.append(len(images))
        # # of batches for one class
        class_batches = min(num_images) // batch_size
        num_test_batches = len(brand_models) * class_batches
        # sometimes database has more data in 'test', some has more in 'unseen'
        if num_batches is not None:
            num_test_batches = min(num_test_batches, num_batches)
            class_batches = round(num_test_batches / len(brand_models))
            print("class batch number of unseen ds is {}".format(class_batches))
            num_test_batches = len(brand_models) * class_batches

        for images in image_paths:
            np.random.shuffle(images)
            img_paths.extend(images[0:class_batches * batch_size])

        return img_paths, num_test_batches


class RocStats(Experiment):
    def __init__(self, params):
        super(RocStats, self).__init__(params)

    def prepare_unseen_dataset(self):
        # form dataset and unseen dataset.
        # both dataset may have different size, might need different batch size.
        unseen_img_paths, self.num_unseen_batches = \
            self.aligned_dataset(self.params.unseen_dataloader.patch_dir, 
                                self.params.unseen_dataloader.brand_models,
                                num_batches=self.num_test_batches,
                                seed=self.random_seed)
        kaggle_img_paths, self.num_kaggle_batches = \
            self.aligned_dataset(self.params.kaggle_dataloader.patch_dir,
                                self.params.kaggle_dataloader.brand_models,
                                num_batches=self.num_test_batches,
                                seed=self.random_seed)
        self.unseen_iter = build_dataset(
                            self.params.unseen_dataloader.patch_dir,
                            self.params.unseen_dataloader.brand_models,
                            "unseen",
                            unseen_img_paths)
        self.kaggle_iter = build_dataset(
                            self.params.kaggle_dataloader.patch_dir,
                            self.params.kaggle_dataloader.brand_models,
                            "kaggle",
                            kaggle_img_paths)

    def prepare_degradation_dataset(self, degradation_id, degradation_factor):
        img_paths = post_processing(self.in_img_paths, 
                                    self.params.experiment.degradation_dir,
                                    degradation_id,
                                    degradation_factor)
        patch_dir = os.path.split(img_paths[0])[0]
        iterator = build_dataset(patch_dir,
                                self.params.dataloader.brand_models,
                                degradation_id,
                                img_paths)
        return iterator

    def experiment(self, model):
        
