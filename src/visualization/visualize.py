# play with the training set
import pandas as pd

from config.config import config

train_folder = config.feature_config['train']
train = pd.read_csv(train_folder, sep=',')

housing = train.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")


