import os
import numpy as np
import tensorflow as tf
from utils.data_preparation import build_dataset, aligned_dataset, post_processing
from utils.log import write_log
from utils.misc import instantiate

def roc_stats(params):

    msg = "... Preparing dataset for statistics experiment\n"
    write_log(params.log.train_log_file, msg)

    for b, m in zip(params.dataloader.brands, 
                    params.dataloader.models):
        params.dataloader.brand_models.append("_".join([b, m]))
    for b, m in zip(params.unseen_dataloader.brands, 
                    params.unseen_dataloader.models):
        params.unseen_dataloader.brand_models.append("_".join([b, m]))
    params.kaggle_dataloader.brand_models = os.listdir(
                        params.kaggle_dataloader.database_image_dir)

    unseen_dataloader = instantiate("dataloader)lib", 
                    params.unseen_dataloader.name)(params)
    kaggle_dataloader = instantiate("dataloader_lib", 
                    params.kaggle_dataloader.name)(params)
    unseen_dataloader.load_data()
    kaggle_dataloader.load_data()






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