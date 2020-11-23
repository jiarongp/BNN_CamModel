import argparse
import os
import json
import time
import importlib
from types import SimpleNamespac

def get_args():
    """
    get params path
    Return:
        args: arguments of the command
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-p', '--params', dest='params', 
                            metavar='P', default='None', 
                            help='the parameters file')
    args = argparser.parse_args()
    return args

def get_params(json_file):
    """
    Get params from json file
    Args:
        json_file: path of the json file
    Return:
        params: parameters from the json file (dict)
    """
    with open(json_file, 'r') as params_file:
        try:
            params = json.load(params_file, object_hook=lambda d: SimpleNamespace(**d))
        except ValueError as err:
            print("... Invalid json: {}".format(err))
            return -1
    return params

def instantiate(module, cls):
    """
    instantiate the class from a certain module specified in the json file.
    Args:
        module: python module
        cls: class in python module
    Return:
        cls_instance: class instance
    """
    try:
        print("... importing " + module)
        module = importlib.import_module(module)
        cls_instance = getattr(module, cls)
        print(cls_instance)
    except Exception as err:
        print("!!! Error creating: {0}".format(err))
        exit(-1)
    return cls_instance

def write_log(log_file, msg):
    with open(log_file, 'a') as f:
        f.write(msg)
        print(msg)