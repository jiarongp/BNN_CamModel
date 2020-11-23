#!/bin/bash
VANILLA=~/BNN_CamModel/params/vanilla_cnn.json
python main.py -p $VANILLA

BNN=~/BNN_CamModel/params/bayesian_cnn.json
python main.py -p $BNN

ENSEMBLE=~/BNN_CamModel/params/ensemble_cnn.json
python main.py -p $ENSEMBLE

EXPERIMENT=~/BNN_CamModel/params/experiment.json
python main.py -p $EXPERIMENT