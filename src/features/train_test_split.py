# -*- coding: utf-8 -*-
import logging
import os

import luigi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

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
    column_strat = config.feature_config['column_strat']

    def requires(self):
        return GetData()

    def output(self):
        return {'train': luigi.LocalTarget(self.train_folder),
                'test': luigi.LocalTarget(self.test_folder)}

    def run(self):

        def build_test_set(self, data, test_size=self.test_size, random_state=self.random_state):
            """

            """
            logger.info('build test and train set with test_size: {}'.format(test_size))

            train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)

            return train_set, test_set

        def stratified_sampling(self, data, column_strat, test_size=self.test_size,
                                random_state=self.random_state):
            """

            """
            logger.info('build test and train set with stratified_sampling & test_size: {}'.format(test_size))
            logger.info('use column: {}'.format(column_strat))

            data["column_cat"] = np.ceil(data[column_strat] / 1.5)
            data["column_cat"].where(data["column_cat"] < 5, 5.0, inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            for train_index, test_index in split.split(data, data['column_cat']):
                strat_train_set = data.loc[train_index]
                strat_test_set = data.loc[test_index]

            # remove cat_attribute
            for set_ in (strat_train_set, strat_test_set):
                set_.drop("column_cat", axis=1, inplace=True)

            return strat_train_set, strat_test_set


        data = pd.read_csv(self.csv_path, sep=',')

        # train, test = build_test_set(self)

        train, test = stratified_sampling(self, data, self.column_strat)

        logger.info('Save test and train set')
        print('Save test and train set')

        train.to_csv(self.train_folder, sep=',')
        test.to_csv(self.test_folder, sep=',')


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=TrainTestSplit)
