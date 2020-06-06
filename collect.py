import os
import urllib
import pandas as pd
import numpy as np
import params
from tqdm import tqdm
from skimage.util.shape import view_as_blocks
from skimage import io

def download(data, images_dir):
    """Download data from the input csv to specific directory
    Args:
        data: a csv file storing the dataset with filename, model, brand and etc.
        images_dir: target root directory for the downloaded images
        csv_path: the path of the csv that stores the brand_model and the corresponding path

    Return:
        a list of brand_model name of the dataset
    """
    csv_rows = []
    path_list = []
    brand_model_list = []

    for i in range((data.shape[0])): 
        csv_rows.append(list(data.iloc[i, :]))

    count = 0
    for csv_row in tqdm(csv_rows):
        filename, brand, model = csv_row[0:3]
        url = csv_row[-1]
        image_path = os.path.join(images_dir, filename)

        try:
            if not os.path.exists(image_path):
                print('Downloading {:}'.format(filename))
                urllib.request.urlretrieve(url, image_path)
            # Load the image and check its dimensions
            img = io.imread(image_path)
            if img is None or not isinstance(img, np.ndarray):
                print('Unable to read image: {:}'.format(filename))
                # removes (deletes) the file path
                os.unlink(image_path)
            # if the size of all images are not zero, then append to the list
            if all(img.shape[:2]):
                count += 1
                brand_model = '_'.join([brand, model])
                brand_model_list.append(brand_model)
                path_list.append(filename)
            else:
                print('Zero-sized image: {:}'.format(filename))
                os.unlink(image_path)

        except IOError:
            print('Unable to decode: {:}'.format(filename))
            os.unlink(image_path)

        except Exception as e:
            print('Error while loading: {:}'.format(filename))
            if os.path.exists(image_path):
                os.unlink(image_path)

    print('Number of images: {:}'.format(len(path_list)))
    return path_list, brand_model_list

# Create output folder if needed
for path in [params.dresden_images_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

data = pd.read_csv(params.dresden)
# train the model on Canon_Ixus70, Nikon_D200, Olympus_MJU
data = data[([m in params.models for m in data['model']])]
images, brand_models = download(data, params.dresden_images_dir)