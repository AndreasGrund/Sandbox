# -*- coding: utf-8 -*-
import os
import json
import logging
import tarfile
from six.moves import urllib
import pandas as pd
import luigi

# logging
current_dir = os.path.dirname(__file__)
logger = logging.getLogger('get_data')
logging.basicConfig(filename=os.path.join(os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'logs'), 'get_data'),level=logging.DEBUG)


class GetData(luigi.ExternalTask):

    # load config
    config_dir = os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), 'config')
    data_config = os.path.join(config_dir, 'data_import.json')
    with open(data_config) as data_file:
        data = json.load(data_file)

    # set up parameter
    _RAW = data['store_raw_data']
    _PATH = os.path.join(os.path.abspath(os.path.join(current_dir, "../..")), _RAW)
    _Project = data['project_name']
    _Project_Path = os.path.join(_PATH, _Project)
    _FILE = data['csv_file']
    _file = data['raw_file']
    _URL = data['url']
    csv_path = os.path.join(_Project_Path, _FILE)

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(self.csv_path)

    def run(self):

        def fetch_data(self):
            """

            """
            logger.info('fetch data from web: {}'.format(self._URL))
            print('fetch data from web: {}'.format(self._URL))

            if not os.path.isdir(self._PATH):
                os.makedirs(self._PATH)
            tgz_path = os.path.join(self._PATH, self._FILE)
            urllib.request.urlretrieve(self._URL, tgz_path)
            file_tgz = tarfile.open(tgz_path)
            file_tgz.extractall(path=self._PATH)
            file_tgz.close()


        def load_local_csv(self):
            """

            """
            logger.info('load local data from {}'.format(self._Project_Path))

            fetch_data()

            print('load local data from {}'.format(self._Project_Path))

            return pd.read_csv(self.csv_path)


if __name__ == "__main__":
    luigi.run(main_task_cls=GetData)
