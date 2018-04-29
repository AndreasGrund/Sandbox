# -*- coding: utf-8 -*-
import logging
import os

import luigi
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from config.config import config
from features.data_cleaning import DataCleaning

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('pipeline')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),
                                          'pipeline'), level=logging.DEBUG)


class ModelTrain(luigi.Task):

    data_prepared = config.feature_config["data_prepared"]
    train_labels = config.feature_config["train_labels"]
    grid_search = config.grid_search["params"]

    def requires(self):
        return DataCleaning()

    def output(self):
        return luigi.LocalTarget("Random_Forest.pkl")

    def run(self):

        # load data
        train = np.loadtxt(self.data_prepared, delimiter=',')
        train = train.reshape((16512, 16))

        train_labels = np.loadtxt(self.train_labels, delimiter=',', usecols=1)

        # Define model
        tree_reg = RandomForestRegressor()

        logger.info("Running grid search for %s", 'Random Forest')

        # ToDo: add Randomized Search
        grid_search = GridSearchCV(tree_reg, self.grid_search, cv=5, scoring="neg_mean_squared_error")

        grid_search.fit(train, train_labels)

        logger.info('performance Random_Forest:')
        logger.info('best parameters: %s', grid_search.best_params_)
        logger.info('best model: %s', grid_search.best_estimator_)

        # ToDO: add information about feature importance
        logger.info('feature importance: %s', grid_search.best_estimator_.feature_importances_)

        # save model
        joblib.dump(grid_search.best_estimator_, "Random_Forest.pkl")


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=ModelTrain)
