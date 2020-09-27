import os
import numpy as np
import tensorflow as tf
from utils.data_preparation import build_dataset, post_processing
from utils.log import write_log
from utils.misc import instantiate
from experiment_lib import SoftmaxStats, MCStats

def experiment(params):
    msg = "... Preparing dataset for statistics experiment\n"
    write_log(params.log.log_file, msg)

    for b, m in zip(params.dataloader.brands, 
                    params.dataloader.models):
        params.dataloader.brand_models.append("_".join([b, m]))
    for b, m in zip(params.unseen_dataloader.brands, 
                    params.unseen_dataloader.models):
        params.unseen_dataloader.brand_models.append("_".join([b, m]))
    params.kaggle_dataloader.brand_models = os.listdir(
                        params.kaggle_dataloader.database_image_dir)

    unseen_dataloader = instantiate("dataloader_lib", 
                    params.unseen_dataloader.name)(params)
    kaggle_dataloader = instantiate("dataloader_lib", 
                    params.kaggle_dataloader.name)(params)
    unseen_dataloader.load_data()
    kaggle_dataloader.load_data()

    if params.experiment.softmax_stats:
        model = instantiate("model_lib",
                    params.softmax_stats.model)(params)
        softmax_stats = SoftmaxStats(params, model)
        softmax_stats.experiment()

    if params.experiment.mc_stats:
        examples_per_epoch = 0
        for m in params.dataloader.brand_models:
            examples_per_epoch += len(os.listdir(os.path.join(
                                    params.dataloader.patch_dir, 
                                    "train", m)))
        model = instantiate("model_lib",
                    params.mc_stats.model)(params, examples_per_epoch)
        mc_stats = MCStats(params, model)
        mc_stats.experiment()