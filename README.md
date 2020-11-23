# Camera modle identification based on Bayesian Neural Network

## Environment
The environment is built successfully with [Anaconda](https://www.anaconda.com), so we recommend to use it.

Alterternatively, you can use virtualenv and pip.
```bash
$ python3 -m venv venv
$ source ./venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## File Structure

### Directories

```
.
├── ckpts
├── data
├── logs
├── params
├── results
└── utils
```
- `ckpts` saves the checkpoints for the trained model, you can use it for evaluation or restore your training.
- `data` stores the data from `Dresden` and `Kaggle` database and their csv files. 
  - name of database is prefix.
  - `base` suffix contains the patches for `train`, `val` and `test`.
  - If you set the `even_database` to `True`, it will generate files with `even` as suffix.
- `logs` is the tensorboard directory.
- `params` store the configuration file for the network training and experiments.
- `results` stores the results, including log file of training and evaluation, as well as images for visulization.
- `utils` stores utility functions.

### Python files
```
├── main.py
├── model_lib.py
├── dataloader_lib.py
├── train.py
├── trainer_lib.py
├── experiment.py
└── experiment_lib.py
```
- `main.py` loads the parameters in configuraion files and runs the program.
- `model_lib` defines model architectures.
- `dataloader_lib` defines dataloader to collect and load images from different dataset, it also includes function like split dataset and extract patches from images.
  - after these, the structure of directory `data` looks like the following:
    ```
    ├── data
    │   ├── *_base
    |   |   ├── test
    |   |   │   ├── Camera_1
    |   |   │   └── Camera_2
    |   |   ├── train
    |   |   │   ├── Camera_1
    |   |   │   └── Camera_2
    |   |   └── val
    |   |   │   ├── Camera_1
    |   |   │   └── Camera_2
    │   ├── database_name
    |   |   │   ├── Camera_1
    |   |   │   └── Camera_2
    │   └── database_name.csv
    ```
- `train.py` builds the model and data iterators, then performs training and evaluation (optional).
- `trainer_lib` provides different training schemes for different models.
- `experiment.py` loads data and performs different experiments.
- `experiment_lib.py` provides different experiment settings.

Utility functions:
```
├── data_preparation.py
├── patch.py
├── misc.py
└── visualization.py
```
- `data_preparation.py` contains the functions that are used for decoding images building data iterator and adding post-processing effects to the images.
- `patch.py` provides functions to divide a image into patches.
- `misc.py` contains functions to parse arguements from command line, instantiate class specified in configuration files and write information to log file.
- `visualization.py` provides function to plot histograms of predictions, ROC curve and also the histograms of weights in different layes.


## Before Running
Set the parameters in the json files under the directory `params`.

Some parameters are worthed to notice:

```json
"run": {
    "name": "VanillaCNN",
    "train": true,
    "evaluate": true,
    "experiment": false
}
```
- `train` and `evaluate` are boolean values, you can change it to enable/disable. 
- `experiment` can be only set true when running the `experiment.json`.

```json
"dataloader": {
    "name": "DresdenDataLoader",
    "database_image_dir": "data/dresden",
    "patch_dir": "data/dresden_base",
    "brands": ["Canon", "Canon", "Nikon", "Nikon", "Sony"],
    "models": ["Ixus70", "Ixus55", "D200", "D70", "DSC-H50"],
    "even_database": false,
}
```
- `name` specify the class we want to use in `dataloader_lib`.
- `database_image_dir` define the path to store the downloaded images from dataset.
- `brands` and `models` are the brand and model information of the camera models, they should be with same size and same order.
- `even_database` is to specify whether to enforce the dataset to be even for each class or not.

## Run
```bash
$ bash run.sh
```
or you can run single file via
```bash
$ python main.py -p $PATH_OF_JSON_FILE
```