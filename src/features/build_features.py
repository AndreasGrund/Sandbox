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
logger = logging.getLogger('pipeline')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),
                                          'pipeline'), level=logging.DEBUG)


class TrainTestSplit(luigi.Task):

    test_size = config.feature_config["test_size"]
    random_state = config.feature_config["random_state"]
    csv_path = config.feature_config["csv_file"]
    train_folder = config.feature_config['train']
    test_folder = config.feature_config['test']

    def requires(self):
        return GetData()

    def output(self):
        return {'train': luigi.LocalTarget(self.train_folder),
                'test': luigi.LocalTarget(self.test_folder)}

    def run(self):

        def build_test_set(self, test_size=self.test_size, random_state=self.random_state):
            """

            """
            logger.info('build test and train set with test_size: {}'.format(test_size))

            data = pd.read_csv(self.csv_path, sep=',')

            train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)

            return train_set, test_set


        train, test = build_test_set(self)

        logger.info('Save test and train set')
        print('Save test and train set')

        train.to_csv(self.train_folder, sep=',')
        test.to_csv(self.test_folder, sep=',')


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=TrainTestSplit)
