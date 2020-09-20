import os
import numpy as np
from multiprocessing import Pool
from skimage.util.shape import view_as_blocks
from skimage.util import random_noise
from skimage import io, filters, img_as_ubyte


def extract_patch(img_path_ls, ds_id, patch_dir,
                    num_patch, extract_span):
    """call the extract function to extract patches from full-sized image.
    Args:
        img_path_ls: paths of images needed to be split into patches.
        ds_id: dataset id, one of ['train', 'val', 'test']. 
                if it's a '.', it will not has ds_id as parent directory.
        patch_dir: parent directory storing the patches.
        num_patch: number of patches being extracted.
        extract_span: size of the region of image to be extracted.
                        if it's 'adaptive', it means will adaptively extract the
                        patches.
    """
    args_ls = []
    for img_path in img_path_ls:
        args_ls += [{'ds_id':ds_id,
                    'img_path':img_path,
                    'patch_dir':patch_dir,
                    'num_patch': num_patch,
                    'extract_span': extract_span}]
    with Pool() as pool:
        pool.map(extract, args_ls)


def extract(args):
    """extract patches from full-sized image.
    Args:
        ds_id: dataset the image belongs to, 'train', 'val' or 'test'.
        img_path: full paths of the source images.
        patch_dir: the parent directory storing the patches.
        num_patch: number of patches being extracted.
        extract_span: size of the region of image to be extracted.
    """
    if args['extract_span'] == 'adaptive':
        img = io.imread(args['img_path'])
        v_patch_span = img.shape[0] // 256 * 256
        h_patch_span = img.shape[1] // 256 * 256
        h_v_num_patch = min([h_patch_span, v_patch_span])
        args['extract_span'] = h_v_num_patch * 256
        args['num_patch'] = pow(h_v_num_patch, 2)
    # 'train/Agfa_DC-504/Agfa_DC-504_0_1_00.png' for example,
    # last part is the patch idex.
    # Use PNG for losslessly storing images
    out_rel_paths = [os.path.join(args['ds_id'],
                    os.path.split(os.path.dirname(args['img_path']))[-1],
                    os.path.splitext(os.path.split(args['img_path'])[-1])[0]
                    +'_'+'{:02}'.format(patch_idx) + '.png')
                    for patch_idx in range(args['num_patch'])]
    read_img = False
    for path in out_rel_paths:
        out_fullpath = os.path.join(args['patch_dir'], path)
        # if there is no this path, then we have to read images
        if not os.path.exists(out_fullpath):
            read_img = True
            break
    if read_img:
        patches = (patchify(args['img_path'], args['extract_span'])
                    .reshape((-1, 256, 256)))
        for path, patch in zip(out_rel_paths, patches):
            out_fullpath = os.path.join(args['patch_dir'], path)
            # the diretory of the patches images
            out_fulldir = os.path.split(out_fullpath)[0]
            if not os.path.exists(out_fulldir):
                os.makedirs(out_fulldir, exist_ok=True)
            if not os.path.exists(out_fullpath):
                io.imsave(out_fullpath, patch, check_contrast=False)


def patchify(img_path, extract_span): 
    """Separate the full-sized image into 256 x 256 image patches. By default, the full-sized
    images is split into 25 patches.
    Args:
        img_path: the path of the source image.
        extract_span: size of the region of image to be extracted.
    Return:
        patches: 25 patches
    """
    img = io.imread(img_path)
    if img is None or not isinstance(img, np.ndarray):
        raise Exception('Unable to read the image: {:}'.format(img_path))
    center = np.divide(img.shape[:2], 2).astype(int)
    start = np.subtract(center, extract_span/2).astype(int)
    end = np.add(center, extract_span/2).astype(int)
    sub_img = img[start[0]:end[0], start[1]:end[1]]
    sub_img = np.asarray(sub_img)
    patches = view_as_blocks(sub_img[:, :, 1], (256, 256))
    return patches