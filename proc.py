import os
import numpy as np
import pandas as pd
import warnings
import params
from tqdm import tqdm
from multiprocessing import Pool
from skimage.util.shape import view_as_blocks
from skimage import io
AUTOTUNE = tf.data.experimental.AUTOTUNE

def patchify(img_path, patch_span=params.patch_span):
    """Separate the full-sized image into 256 x 256 image patches. By default, the full-sized
    images is split into 25 patches.
    Args:
        img_path: the path of the source image.
        patch_span: decide the number of patches, which is patch_span^2.
    Return:
        patches: 25 patches
    """
    img = io.imread(img_path)
    if img is None or not isinstance(img, np.ndarray):
        print('Unable to read the image: {:}'.format(img_path))

    center = np.divide(img.shape[:2], 2).astype(int)
    start = np.subtract(center, patch_span/2).astype(int)
    end = np.add(center, patch_span/2).astype(int)
    sub_img = img[start[0]:end[0], start[1]:end[1]]
    sub_img = np.asarray(sub_img)
    patches = view_as_blocks(sub_img[:, :, 1], (256, 256))
    return patches


def patch(path, dataset):
    """call the extract function to extract patches from full-sized image
    Args:
        path: paths for images needed to be split into patches
        dataset: one of ['train', 'val', 'test']
    """
    imgs_list = []
    for img_path in path:
        imgs_list += [{'dataset':dataset,
                    'img_path':img_path
                    }]
    num_processes = 8
    pool = Pool(processes=num_processes)
    pool.map(extract, imgs_list)


def extract(args):
    """extract patches from full-sized image
    Args:
        dataset: dataset the image belongs to, 'train', 'val' or 'test'
        img_path: full paths of the source images
    Return:
        output_rel_paths: the paths of extracted patches. For example:
                          'train/brand_model/filename_idx.png'
    """
    # 'train/Agfa_DC-504/Agfa_DC-504_0_1_00.png' for example,
    # last part is the patch idex.
    # Use PNG for losslessly storing images
    output_rel_paths = [os.path.join(args['dataset'],
                        os.path.split(os.path.dirname(args['img_path']))[-1],
                        os.path.splitext(os.path.split(args['img_path'])[-1])[0]+'_'+'{:02}'.format(patch_idx) + '.png')\
                        for patch_idx in range(params.patch_num)]
    read_img = False
    for out_path in output_rel_paths:
        out_fullpath = os.path.join(params.patches_dir, out_path)

        # if there is no this path, then we have to read images
        if not os.path.exists(out_fullpath):
            read_img = True
            break
    if read_img:
        patches = patchify(args['img_path']).reshape((-1, 256, 256))
    
        for out_path, patch in zip(output_rel_paths, patches):
            out_fullpath = os.path.join(params.patches_dir, out_path)
            # the diretory of the patches images
            out_fulldir = os.path.split(out_fullpath)[0]
            if not os.path.exists(out_fulldir):
                os.makedirs(out_fulldir)
            if not os.path.exists(out_fullpath):
                io.imsave(out_fullpath, patch, check_contrast=False)
    return output_rel_paths