
from sklearn.model_selection import train_test_split
import os
import json
import logging

from data.get_data import GetData

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('build_features')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),'build_feature'), level=logging.DEBUG)


class TrainTestSplit:

    # load config
    current_dir = os.path.dirname(__file__)
    config_dir = os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'config')
    data_config = os.path.join(config_dir, 'data_import.json')
    with open(data_config) as data_file:
        data = json.load(data_file)

    # define parameter
    _RAW = data['store_raw_data']
    _PATH = os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), _RAW)
    _Project = data['project_name']
    _Project_Path = os.path.join(_PATH, _Project)
    _FILE = data['csv_file']


    def build_test_set(self, test_size=0.2, random_state=42):
        """

        """
        logger.info('build test and train set with test_size: {}'.format(test_size))

        data = GetData().load_local_csv()

        train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)

        return train_set, test_set


train, test = TrainTestSplit().build_test_set()
