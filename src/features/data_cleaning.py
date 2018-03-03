# -*- coding: utf-8 -*-
import logging
import os

import luigi
import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer

from config.config import config
from features.train_test_split import TrainTestSplit

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('pipeline')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),
                                          'pipeline'), level=logging.DEBUG)


class DataCleaning(luigi.Task):

    train_folder = config.feature_config['train']
    test_folder = config.feature_config['test']
    target_value = config.feature_config["target_value"]
    input_strategy = config.feature_config["input_strategy"]

    def requires(self):
        return TrainTestSplit()

    def output(self):
        pass

    def run(self):
        """

        """
        # load training set
        train_raw = pd.read_csv(config.feature_config['train'], sep=',', index_col=0)

        train = train_raw.drop(self.target_value, axis=1)
        train_labels = train_raw[self.target_value].copy()

        def num_inputer():

            inputer = Imputer(strategy=self.input_strategy)

            train_num = train.select_dtypes(include=[np.number])

            inputer.fit(train_num)

            X = inputer.transform(train_num)

            train_num = pd.DataFrame(X, columns=train_num.columns)

            return train_num

        def label_encoder():

            encoder = LabelBinarizer()

            train_cat = train.select_dtypes(include=[np.chararray])

            train_cat_encoded = encoder.fit_transform(train_cat)

            return train_cat_encoded

        test = label_encoder()






if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=DataCleaning)
