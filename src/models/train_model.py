# -*- coding: utf-8 -*-
import logging
import os

import luigi
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
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

    def requires(self):
        return DataCleaning()

    def output(self):
        pass

    def run(self):

        # load data
        train = np.loadtxt(self.data_prepared, delimiter=',')
        train = train.reshape((16512, 16))

        train_labels = np.loadtxt(self.train_labels, delimiter=',', usecols=1)

        # Define model
        tree_reg = RandomForestRegressor()

        scores = cross_val_score(tree_reg, train, train_labels, scoring="neg_mean_squared_error", cv=10)

        tree_rmse_scores = np.sqrt(-scores)

        def display_scores(scores):
            print("Scores: ", scores)
            print("Mean: ", scores.mean())
            print("Standard deviation: ", scores.std())

        display_scores(tree_rmse_scores)

        # save model
        joblib.dump(tree_reg, "Random_Forest.pkl")


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=ModelTrain)
