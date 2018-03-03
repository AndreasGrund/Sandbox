# -*- coding: utf-8 -*-
import logging
import os

import luigi
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import config
from data.get_data import GetData

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('build_features')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),
                                          'build_feature'), level=logging.DEBUG)


class TrainTestSplit(luigi.Task):

    test_size = config.feature_config["test_size"]
    random_state = config.feature_config["random_state"]

    _RAW = config.data_location['store_raw_data']
    _PATH = os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), _RAW)
    _Project = config.data_location['project_name']
    _Project_Path = os.path.join(_PATH, _Project)
    _FILE = config.data_location['csv_file']
    csv_path = os.path.join(_Project_Path, _FILE)

    def requires(self):
        return GetData()

    # def output(self):
    #     csv_path = os.path.join(self._Project_Path, self._FILE)
    #     return luigi.LocalTarget(csv_path)

    def run(self):

        def build_test_set(self, test_size=self.test_size, random_state=self.random_state):
            """

            """
            logger.info('build test and train set with test_size: {}'.format(test_size))

            data = pd.read_csv(self.csv_path, sep=',')

            train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)

            return train_set, test_set


        train, test = build_test_set(self)


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=TrainTestSplit)
