import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSet
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '/kaggle/input/cityscapes/Cityspaces'
    if name == 'gta' or name == 'gtaUniform':
        return '/kaggle/input/gtav-dataset/GTAV'
    if name == 'synthia':
        return '../data/RAND_CITYSCAPES'
