# 'bnn' / 'vanilla'
model_type = 'bnn'
# 'dresden' / 'RAISE'
database = 'dresden'
even_database = True
image_root = 'data/'

if database == 'dresden':
    # dresden brand model
    # brands = ['Canon', 'Canon', 'Nikon', 'Nikon', 'Olympus']
    # models = ['Ixus70', 'Ixus55', 'D200', 'D70', 'mju-1050SW']
    brands = ['Canon', 'Canon', 'Nikon', 'Nikon']
    models = ['Ixus70', 'Ixus55', 'D200', 'D70']
    unseen_brands = ['Agfa', 'Canon', 'Sony']
    unseen_models = ['DC-830i', 'PowerShotA640', 'DSC-W170']
    ds_csv = 'data/dresden.csv'
    ds_image_dir = image_root + 'dresden'
    patch_dir = image_root + ('even_dresden_base'
                if even_database else 'dresden_base')
    unseen_dir = image_root + 'dresden_unseen'
    print_fig_step = 10

elif database == 'RAISE':
    # RAISE brand model
    brands = ['Nikon', 'Nikon']
    models = ['D90', 'D7000']
    # unseen can be changed here
    unseen_brands = ['Nikon']
    unseen_models = ['D40']
    ds_csv = 'data/RAISE_2k.csv'
    ds_image_dir = image_root + 'RAISE'
    patch_dir = image_root + ('even_RAISE_base' 
                if even_database else 'RAISE_base')
    unseen_dir = image_root  + 'RAISE_unseen'
    print_fig_step = 10

brand_models = ['_'.join([b, m]) for (b, m) in zip(brands, models)]
unseen_brand_models = ['_'.join([b, m]) for (b, m) in zip(unseen_brands, unseen_models)]

IMG_HEIGHT = 256
IMG_WIDTH = 256
patch_num = 25
patch_span = 256 * 5
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_CLASSES = len(brand_models)
num_monte_carlo = 20
patience = 5
# restore training
restore = False

HParams = {'init_learning_rate':0.0001,
           'init_prior_scale_mean':-1.9994,
           'init_prior_scale_std':-0.30840, 
           'std_prior_scale':3.4210}

