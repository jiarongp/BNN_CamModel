# Camera modle identification based on Bayesian Neural Network

## Environment

```bash
$ python3 -m venv venv
$ source ./venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## File Structure

Directories

```
├── ckpts
├── data
├── examples
├── logs
└── results
```
- `ckpts` saves the checkpoints for the trained model, you can use it for evaluation or restore your training.
- `data` stores the data from `dresden` and `RAISE` database and there csv files. 
  - name of database is prefix.
  - `base` suffix contains the patches for `train`, `val` and `test`.
  - If you set the `even_database` to `True`, it will generate files with `even` as suffix.
- `logs` is the tensorboard directory
- `results` stores the results from `train.py`, `evaluate.py` and `stats.py`

Python files
```
├── data_preparation.py
├── model_lib.py
├── train.py
├── evaluate.py
├── params.py
├── stats.py
└── utils.py
```
- `data_preparation.py` contains the functions that are used for downloading images from database, splitting images into `train`, `val` and `test` dataset and extract patches from these full-sized images into `data\*_base` directory.
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
- `model_lib.py` defines the model, including a vanilla CNN and a bayesian CNN.
- `train.py` defines the training loop.
- `evaluate.py` defines the evaluation.
- `params.py` stores the paramters like specific directories and training parameters.
- `stats.py` calculate statics of the model. e.g. ROC curves and [softmax statistics](https://github.com/hendrycks/error-detection)
- `utils.py` contains useful functions

## Before Running

Set the parameter in the `params.py`

```python
model_type = 'vanilla' 
database = 'RAISE'
even_database = True
image_root = 'data/'
```

- `model_type` has either `'vanilla'` or `'bnn'` as input to set either the running on vanilla CNN or Bayesian CNN.
- `database` has either `'dresden'` or `'RAISE'` as input to set either running on dresden or RAISE database.
- `even_database` force the data have the same number of images for each classes.
- `images_root` set the root directory of the image data.

## Training

```bash
$ python train.py
```
## Evaluation

```bash
$ python evaluate.py
```

## Statistics

```bash
$ python stats.py
```