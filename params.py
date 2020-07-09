brands = ['Canon', 'Nikon', 'Olympus']
models = ['Ixus70', 'D200', 'mju-1050SW']
brand_models = ['_'.join([b, m]) for (b, m) in zip(brands, models)]

unseen_brands = ['Agfa', 'Canon', 'Canon', 'Nikon', 'Sony']
unseen_models = ['DC-830i', 'Ixus55', 'PowerShotA640', 'D70', 'DSC-W170']
unseen_brand_models = ['_'.join([b, m]) for (b, m) in zip(unseen_brands, unseen_models)]

dresden = 'data/dresden.csv'
dresden_images_dir = 'data/dresden'
unseen_images_dir = 'data/unseen/test'

IMG_HEIGHT = 256
IMG_WIDTH = 256
patch_num = 25
patch_span = 256 * 5
patches_dir = 'data/base'

BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_CLASSES = len(brand_models)
num_monte_carlo = 30
patience = 3

HParams = {'init_learning_rate':0.001,
           'init_prior_scale_mean':-1.9994,
           'init_prior_scale_std':-0.30840, 
           'std_prior_scale':3.4210}

