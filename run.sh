#!/bin/bash

#PARAMSFILE=~/BNN_CamModel/params/vanilla_cnn.json
#PARAMSFILE=~/BNN_CamModel/params/bayesian_cnn.json
#PARAMSFILE=~/BNN_CamModel/params/ensemble_cnn.json
PARAMSFILE=~/BNN_CamModel/params/experiment.json
python main.py -p $PARAMSFILE

