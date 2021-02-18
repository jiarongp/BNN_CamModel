import os
import numpy as np
import tensorflow as tf
from utils.data_preparation import build_dataset, post_processing
from utils.misc import instantiate, write_log
from experiment_lib import SoftmaxStats, MCStats, MultiMCStats, EnsembleStats, MCDegradationStats

def experiment(params):
    """
    perfome experiments.
    """
    msg = "... Preparing dataset for statistics experiment\n"
    write_log(params.log.log_file, msg)
    
    # prepare unseen data from Dresden and Kaggle dataset
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

    # softmax statistics and ensemble use the vanilla CNN
    if params.experiment.softmax_stats or params.experiment.ensemble_stats:
        model = instantiate("model_lib",
                    params.softmax_stats.model)(params)
        if params.experiment.softmax_stats:
            softmax_stats = SoftmaxStats(params, model)
            softmax_stats.experiment()
        if params.experiment.ensemble_stats:
            ensemble_stats = EnsembleStats(params, model)
            ensemble_stats.experiment()

    # other experiments use the BNN.
    if params.experiment.mc_stats or params.experiment.multi_mc_stats or params.experiment.mc_degradation_stats:
        examples_per_epoch = 0
        for m in params.dataloader.brand_models:
            examples_per_epoch += len(os.listdir(os.path.join(
                                    params.dataloader.patch_dir, 
                                    "train", m)))
        model = instantiate("model_lib",
                    params.mc_stats.model)(params, examples_per_epoch)
        if params.experiment.mc_stats:
            mc_stats = MCStats(params, model)
            mc_stats.experiment()
        if params.experiment.multi_mc_stats:
            multi_mc_stats = MultiMCStats(params, model)
            multi_mc_stats.experiment()
        if params.experiment.mc_degradation_stats:
            mc_degradation_stats = MCDegradationStats(params, model)
            mc_degradation_stats.experiment()