import argparse
import os
import json
import time
from types import SimpleNamespace


def get_args():
    """
    get params path
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-p', '--params', dest='params', 
                            metavar='P', default='None', 
                            help='The Parameters File')

    args = argparser.parse_args()
    return args

def get_params(json_file):
    """
    Get params from json file
    Args:
        json_file: path of the json file
    Return:
        params (dict)
    """
    with open(json_file, 'r') as params_file:
        try:
            params = json.load(params_file, object_hook=lambda d: SimpleNamespace(**d))
        except ValueError as err:
            print("... Invalid json: {}".format(err))
            return -1
    return params
