# -*- coding: utf-8 -*-
'''''Configuration of directories, target column names, folds'''
import os


class ParamConfig:
    def __init__(self):

        self.project_folder = {"folder": os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))}

        self.data_location = {
                      "project_name": "housing",
                      "url": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz",
                      "csv_file": "housing.csv",
                      "store_raw_data": "data/raw",
                      "raw_file": "housing.tgz"
                                }

        self.feature_config = {
                    "test_size": 0.2,
                    "random_state": 42,
                    "csv_file": os.path.join(self.project_folder["folder"],
                    self.data_location['store_raw_data'], self.data_location['project_name'],
                    self.data_location['csv_file']),
                    "test": os.path.join(self.project_folder["folder"], 'data/interim/test.txt'),
                    "train": os.path.join(self.project_folder['folder'], 'data/interim/train.txt'),
                    "column_strat": "median_income",
                    "target_value": "median_house_value",
                    "input_strategy": "median",
                    "data_prepared": os.path.join(self.project_folder["folder"], 'data/processed/train_prepared.txt'),
                    "train_labels": os.path.join(self.project_folder["folder"], 'data/processed/train_labels.txt')

        }


# initialize a param config
config = ParamConfig()
