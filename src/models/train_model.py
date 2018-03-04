# -*- coding: utf-8 -*-
import logging
import os

import luigi
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

        linear_reg = LinearRegression()
        linear_reg.fit(train, train_labels)

        predictions = linear_reg.predict(train)

        linear_mse = mean_squared_error(train_labels, predictions)
        linear_rmse = np.sqrt(linear_mse)

        print(linear_rmse)


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=ModelTrain)
