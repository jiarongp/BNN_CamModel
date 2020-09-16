# 'bnn' / 'vanilla'
model_type = 'bnn'
# 'dresden' / 'RAISE'
database = 'dresden'
even_database = False
image_root = 'data/'

if database == 'dresden':
    # 6 models
    # brands = ['Canon', 'Canon', 'Nikon', 'Nikon', 'Sony', 'Agfa']
    # models = ['Ixus70', 'Ixus55', 'D200', 'D70', 'DSC-H50', 'DC-733s']
    # 5 models
    brands = ['Canon', 'Canon', 'Nikon', 'Nikon', 'Sony']
    models = ['Ixus70', 'Ixus55', 'D200', 'D70', 'DSC-H50']
    # 4 models
    # brands = ['Canon', 'Canon', 'Nikon', 'Nikon']
    # models = ['Ixus70', 'Ixus55', 'D200', 'D70']
    # 3 models
    # brands = []
    # models = []

    unseen_brands = ['Agfa', 'Canon', 'Sony', 'Samsung', 'Nikon']
    unseen_models = ['DC-830i', 'PowerShotA640', 'DSC-W170', 'L74wide', 'CoolPixS710']
    ds_csv = 'data/dresden.csv'
    ds_image_dir = image_root + 'dresden'
    patch_dir = image_root + ('even_dresden_base'
                if even_database else 'dresden_base')
    unseen_dir = image_root + 'dresden_unseen'
    kaggle_dir = image_root + 'kaggle_unseen'
    print_fig_step = 50


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
adaptive_span = True
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_CLASSES = len(brand_models)
num_monte_carlo = 10
patience = 5
# restore training
restore = False

HParams = {'init_learning_rate':0.001,
           'init_prior_scale_mean':-1,
           'init_prior_scale_std':.1, 
           'std_prior_scale':1.5}

