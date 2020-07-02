brands = ['Canon', 'Nikon', 'Olympus']
models = ['Ixus70', 'D200', 'mju-1050SW']
brand_models = ['_'.join([b, m]) for (b, m) in zip(brands, models)]
# label_to_num = {}
# for i in range(len(brand_models)):
#     label_to_num[brand_models[i]] = i
dresden = 'data/dresden.csv'
dresden_images_dir = 'data/dresden'

IMG_HEIGHT = 256
IMG_WIDTH = 256
patch_num = 25
patch_span = 256 * 5
patches_dir = 'data/base'

BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = len(brand_models)
num_monte_carlo = 30

HParams = {'init_learning_rate':0.001,
           'init_prior_scale_mean':-1.9994,
           'init_prior_scale_std':-0.30840, 
           'std_prior_scale':3.4210}

