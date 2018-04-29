# -*- coding: utf-8 -*-
import logging
import os

import luigi

from sklearn.externals import joblib

from config.config import config
from models.train_model import ModelTrain
from features.data_cleaning import DataCleaningTest

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('pipeline')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),
                                          'pipeline'), level=logging.DEBUG)

class ModelTest(luigi.Task):

    data_prepared = config.feature_config["data_prepared"]
    train_labels = config.feature_config["train_labels"]

    def requires(self):
        return ModelTrain()

    def output(self):
        pass

    def run(self):

        # load model
        best_model = joblib.load("Random_Forest.pkl")




if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=ModelTest)
