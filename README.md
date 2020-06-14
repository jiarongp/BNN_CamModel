# Camera modle identification based on Bayesian Neural Network

## File Structure

```
├── data_preparation.py
├── model_lib.py
├── train.py
├── evaluate.py
├── params.py
└── utils.py
```
- `data_preparation.py` contains the functions that are used for downloading images from `data/dresden.csv`, splitting images into `train`, `val` and `test` dataset and extract patches from these full-sized images into `data\base` directory.
  - after these, the structure of directory `data` looks like the following:
    ```
    ├── data
    │   ├── base
    |   |   ├── test
    |   |   │   ├── Canon_Ixus70
    |   |   │   └── Nikon_D200
    |   |   ├── train
    |   |   │   ├── Canon_Ixus70
    |   |   │   └── Nikon_D200
    |   |   └── val
    |   |       ├── Canon_Ixus70
    |   |       └── Nikon_D200
    │   ├── dresden
    │   │   ├── Canon_Ixus70
    │   │   └── Nikon_D200
    │   └── dresden.csv
    ```
- `model_lib.py` defines the model.
- `train.py` defines the training loop.
- `evaluate.py` defines the evaluation.
- `params.py` stores the paramters like specific directories and training parameters.
- `utils.py` contains useful functions
