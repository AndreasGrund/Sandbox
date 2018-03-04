# -*- coding: utf-8 -*-
import logging
import os

import luigi
import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from config.config import config
from features.train_test_split import TrainTestSplit

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('pipeline')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),
                                          'pipeline'), level=logging.DEBUG)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


class DataCleaning(luigi.Task):

    train_folder = config.feature_config['train']
    test_folder = config.feature_config['test']
    target_value = config.feature_config["target_value"]
    input_strategy = config.feature_config["input_strategy"]
    data_prepared = config.feature_config["data_prepared"]
    train_labels = config.feature_config["train_labels"]

    def requires(self):
        return TrainTestSplit()

    def output(self):
        return {'data_prepared': luigi.LocalTarget(self.data_prepared),
                'train_labels': luigi.LocalTarget(self.train_labels)}

    def run(self):
        """

        """
        # load training set
        train_raw = pd.read_csv(config.feature_config['train'], sep=',', index_col=0)

        train = train_raw.drop(self.target_value, axis=1)
        train_labels = train_raw[self.target_value].copy()

        def num_inputer():

            logger.info('prepare numeric features')
            print('prepare numeric features')

            train_num = train.select_dtypes(include=[np.number])
            num_attribs = list(train_num)

            num_steps = Pipeline([
                ('selector', DataFrameSelector(num_attribs)),
                ('imputer', Imputer(strategy=self.input_strategy)),
                ('attribs_adder', CombinedAttributesAdder()),
                ('std_scaler', StandardScaler()),
            ])

            return num_steps

        def label_encoder():
            logger.info('prepare cat features')
            print('prepare cat features')

            train_cat = train.select_dtypes(include=[np.chararray])

            cat_attribs = list(train_cat)

            cat_steps = Pipeline([
                ('selector', DataFrameSelector(cat_attribs)),
                ('label_binarizer', LabelBinarizerPipelineFriendly()),
            ])

            return cat_steps

        def full_pipeline():
            logger.info('prepare complete feature pipeline')
            print('prepare complete feature pipeline')

            full_pipeline = FeatureUnion(transformer_list=[
                ('num_pipeline', num_pipeline),
                ('cat_pipeline', cat_pipeline),
            ])

            return full_pipeline

        num_pipeline = num_inputer()

        cat_pipeline = label_encoder()

        full_pipeline = full_pipeline()

        data_prepared = full_pipeline.fit_transform(train)

        logger.info('save cleaned train set')
        print('save cleaned train set')
        data_prepared.tofile(self.data_prepared, sep=',')
        train_labels.to_csv(self.train_labels, sep=',')


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=DataCleaning)
