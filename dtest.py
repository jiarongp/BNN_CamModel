import data_preparation as dp
import params
import pandas as pd
import utils

utils.set_logger('results/dtest.log')
# dataset='RAISE'
# ds_csv, ds_dir = ((params.dresden, params.dresden_images_dir) 
#                    if (dataset=='dresden') 
#                    else (params.RAISE, params.RAISE_images_dir))
# data = pd.read_csv(ds_csv)
# if dataset=='dresden':
#     data = data[([m in params.models for m in data['model']])]
# dp.collect_dataset(data, ds_dir,
#                    params.brand_models, 
#                    dataset, download=True)    
dp.collect_split_extract(download_images=False, 
                         dataset='RAISE',
                         parent_dir=params.RAISE_patches)