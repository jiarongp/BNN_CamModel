# dresden brand model
brands = ['Canon', 'Canon', 'Nikon', 'Nikon', 'Olympus', 'Sony', 'Sony']
models = ['Ixus70', 'Ixus55', 'D200', 'D70', 'mju-1050SW', 'DSC-H50', 'DSC-T77']

# # RAISE brand model
# brands = ['Nikon', 'Nikon', 'Nikon']
# models = ['D40', 'D90', 'D7000']
brand_models = ['_'.join([b, m]) for (b, m) in zip(brands, models)]

unseen_brands = ['Agfa', 'Canon', 'Sony']
unseen_models = ['DC-830i', 'PowerShotA640', 'DSC-W170']
unseen_brand_models = ['_'.join([b, m]) for (b, m) in zip(unseen_brands, unseen_models)]

dresden = 'data/dresden.csv'
dresden_images_dir = 'data/dresden'
RAISE = 'data/RAISE_2k.csv'
RAISE_images_dir = 'data/RAISE'

unseen_patches_dir = 'data/unseen/'
unseen_images_dir = 'data/unseen/test'

IMG_HEIGHT = 256
IMG_WIDTH = 256
patch_num = 25
patch_span = 256 * 5
dresden_patches = 'data/dresden_base'
RAISE_patches = 'data/RAISE_base'

BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_CLASSES = len(brand_models)
num_monte_carlo = 30
patience = 5

HParams = {'init_learning_rate':0.001,
           'init_prior_scale_mean':-1.9994,
           'init_prior_scale_std':-0.30840, 
           'std_prior_scale':3.4210}

