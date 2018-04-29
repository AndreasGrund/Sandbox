# -*- coding: utf-8 -*-
import logging
import os

import luigi
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

from config.config import config
from models.train_model import ModelTrain

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('pipeline')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),
                                          'pipeline'), level=logging.DEBUG)


class ModelTest(luigi.Task):

    data_prepared_test = config.feature_config["data_prepared_test"]
    test_labels = config.feature_config["test_labels"]
    final_preds = config.feature_config["final_preds"]

    def requires(self):
        return ModelTrain()

    def output(self):
        return {'final_predictions': luigi.LocalTarget(self.final_preds)}

    def run(self):

        # load model and data
        best_model = joblib.load("Random_Forest.pkl")

        test = np.loadtxt(self.data_prepared_test, delimiter=',')
        test_labels = np.loadtxt(self.test_labels, delimiter=',', usecols=1)
        test = test.reshape((4128, 16))

        final_predictions = best_model.predict(test)

        final_mse = mean_squared_error(test_labels, final_predictions)

        final_rmse = np.sqrt(final_mse)
        logger.info('final model evaluation: %s', final_rmse)

        # save final predictions and test_labels
        final_predictions.tofile(self.final_preds, sep=',')


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=ModelTest)
