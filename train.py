import os
import tensorflow as tf
from utils.log import write_log
from utils.data_preparation import build_dataset
from utils.misc import instantiate
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def train(params):
    msg = "... Preparing dataset for training\n"
    write_log(params.log.train_log_file, msg)
    for b, m in zip(params.dataloader.brands, 
                            params.dataloader.models):
        params.dataloader.brand_models.append("_".join([b, m]))
    # collect & split in to train, val and test & extract to patches
    dataloader = instantiate("dataloader_lib", 
                    params.dataloader.name)(params)
    dataloader.load_data()

    # choose class_imbalance here,
    # using oversampling the minority class, 
    # not sure if it really affect the performance, 
    # since still using the whole training size to calculate the steps
    # resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE) in tensorflow example
    class_imbalance = False if params.dataloader.even_database else True
    train_iter = build_dataset(params.dataloader.patch_dir,
                                params.dataloader.brand_models,
                                'train', params.trainer.batch_size,
                                class_imbalance=class_imbalance)
    val_iter = build_dataset(params.dataloader.patch_dir, 
                            params.dataloader.brand_models,
                            'val', params.trainer.batch_size)
    test_iter = build_dataset(params.dataloader.patch_dir, 
                            params.dataloader.brand_models,
                            'test', params.evaluate.batch_size)
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
    trainer = instantiate("trainer_lib",
                            params.trainer.name)(params, model)
    trainer.train(train_iter, val_iter)
    trainer.evaluate(test_iter)