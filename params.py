brands = ['Canon', 'Nikon']
models = ['Ixus70', 'D200']
brand_models = ['_'.join([b, m]) for (b, m) in zip(brands, models)]
dresden = 'data/dresden.csv'
dresden_images_dir = 'data/dresden'

IMG_HEIGHT = 256
IMG_WIDTH = 256
patch_num = 25
patch_span = 256 * 5
patches_dir = 'data/base'

BATCH_SIZE = 64
