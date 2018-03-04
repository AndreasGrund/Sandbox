# -*- coding: utf-8 -*-
import logging
import os
import tarfile

import luigi
import pandas as pd
from six.moves import urllib

from config.config import config

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('pipeline')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'),
                                          'pipeline'), level=logging.DEBUG)


class GetData(luigi.Task):

    # set up parameter
    _RAW = config.data_location['store_raw_data']
    _PATH = os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), _RAW)
    _Project = config.data_location['project_name']
    _Project_Path = os.path.join(_PATH, _Project)
    _FILE = config.data_location['csv_file']
    _file = config.data_location['raw_file']
    _URL = config.data_location['url']

    def requires(self):
        pass

    def output(self):
        csv_path = os.path.join(self._Project_Path, self._FILE)
        return luigi.LocalTarget(csv_path)

    def run(self):

        def fetch_data(url=self._URL, path=self._Project_Path, file=self._file):
            """

            """
            logger.info('fetch data from web: {}'.format(url))
            print('fetch data from web: {}'.format(url))

            if not os.path.isdir(path):
                os.makedirs(path)
            tgz_path = os.path.join(path, file)
            urllib.request.urlretrieve(url, tgz_path)
            file_tgz = tarfile.open(tgz_path)
            file_tgz.extractall(path=path)
            file_tgz.close()

        logger.info('load local data from {}'.format(self._PATH))

        fetch_data()

        print('load local data from {}'.format(self._PATH))

        csv_path = os.path.join(self._Project_Path, self._FILE)
        return pd.read_csv(csv_path, sep=',')


if __name__ == "__main__":
    luigi.run(["--local-scheduler"], main_task_cls=GetData)
