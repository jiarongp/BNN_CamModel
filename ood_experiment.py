import pandas as pd
import params
import data_preparation
from tqdm import trange 

# collect odd data
data = pd.read_csv(params.dresden)
data = data[([m in params.unseen_models for m in data['model']])]
image_paths = data_preparation.collect_dataset(data, 
                                               params.dresden_images_dir,
                                               params.unseen_brand_models,
                                               download=False)

for i in trange(len(image_paths)):
    data_preparation.patch(path=image_paths, dataset='test', parent_dir=params.unseen_patches_dir)
print("... Done\n")