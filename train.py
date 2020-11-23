import os
import tensorflow as tf
from utils.data_preparation import build_dataset
from utils.misc import instantiate, write_log
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def train_eval(params):
    msg = "... Preparing dataset\n"
    write_log(params.log.log_file, msg)
    # collect & split in to train, val and test & extract to patches
    dataloader = instantiate("dataloader_lib", 
                    params.dataloader.name)(params)
    dataloader.load_data()
    # claculate the kl_weight for BNN.
    examples_per_epoch = 0
    for m in params.dataloader.brand_models:
        examples_per_epoch += len(os.listdir(os.path.join(
                                params.dataloader.patch_dir, 
                                "train", m)))
    if params.model.name in ["BayesianCNN", "EB_BayesianCNN"]:
        model = instantiate("model_lib", 
                            params.model.name)(params, examples_per_epoch)
    else:
        model = instantiate("model_lib", 
                    params.model.name)(params)    
    trainer = instantiate("trainer_lib", params.trainer.name)(params, model)

    if params.run.train:
        # if True, the minority class will be oversampled during training.
        # if False, the training set will be enforce to have the same amount of data for each class.
        class_imbalance = False if params.dataloader.even_database else True
        train_iter = build_dataset(params.dataloader.patch_dir,
                                    params.dataloader.brand_models,
                                    'train', params.trainer.batch_size,
                                    class_imbalance=class_imbalance)
        val_iter = build_dataset(params.dataloader.patch_dir, 
                                params.dataloader.brand_models,
                                'val', params.trainer.batch_size)
        trainer.train(train_iter, val_iter)

    if params.run.evaluate:
        test_iter = build_dataset(params.dataloader.patch_dir,
                                params.dataloader.brand_models,
                                'test', params.evaluate.batch_size)
        trainer.evaluate(test_iter)