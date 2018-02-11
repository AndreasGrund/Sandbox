# -*- coding: utf-8 -*-
import os
import json
import logging
import tarfile
from six.moves import urllib
import pandas as pd

# load config
current_dir = os.path.dirname(__file__)
config_dir = os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'config')
data_config = os.path.join(config_dir, 'data_import.json')
with open(data_config) as data_file:
    data = json.load(data_file)

# set up parameter
_RAW = data['store_raw_data']
_PATH = os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), _RAW)
_Project = data['project_name']
_Project_Path = os.path.join(_PATH, _Project)
_FILE = data['csv_file']
_file = data['raw_file']
_URL = data['url']


def fetch_data(url=_URL, path=_Project_Path, file=_file):
    """

    """
    logger = logging.getLogger(__name__)
    logger.info('fetch data from web: {}'.format(url))
    print('fetch data from web: {}'.format(url))

    if not os.path.isdir(path):
        os.makedirs(path)
    tgz_path = os.path.join(path, file)
    urllib.request.urlretrieve(url, tgz_path)
    file_tgz = tarfile.open(tgz_path)
    file_tgz.extractall(path=path)
    file_tgz.close()


def load_local_csv(path=_Project_Path, file=_FILE):
    """

    """
    logger = logging.getLogger(__name__)
    logger.info('load local data from {}'.format(path))

    fetch_data()

    print('load local data from {}'.format(path))
    csv_path = os.path.join(path, file)
    return pd.read_csv(csv_path)


data = load_local_csv()
