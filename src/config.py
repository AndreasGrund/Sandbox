# -*- coding: utf-8 -*-
'''''Configuration of directories, target column names, folds'''


class ParamConfig:
    def __init__(self):

        self.data_location = {
                      "data_file": "E:/Datasets",
                      "project_name": "housing",
                      "url": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz",
                      "csv_file": "housing.csv",
                      "store_raw_data": "data/raw",
                      "raw_file": "housing.tgz"
                                }

        self.feature_config = {
                                "test_size": 0.2,
                                "random_state": 42
        }


# initialize a param config
config = ParamConfig()
