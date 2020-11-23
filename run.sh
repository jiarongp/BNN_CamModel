#!/bin/bash
VANILLA=params/vanilla_cnn.json
python main.py -p $VANILLA

BNN=params/bayesian_cnn.json
python main.py -p $BNN

ENSEMBLE=params/ensemble_cnn.json
python main.py -p $ENSEMBLE

EXPERIMENT=params/experiment.json
python main.py -p $EXPERIMENT
