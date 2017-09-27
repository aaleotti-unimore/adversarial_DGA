import logging
import os
import random as rn
import sys
import tensorflow as tf
import keras as K
import matplotlib
import numpy as np
from sklearn.utils import shuffle
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
sys.path.append("../detect_DGA")

from sklearn.model_selection import train_test_split
from model import Model, pierazzi_baseline, pierazzi_normalized_baseline, verysmall_baseline
from features.data_generator import load_features_dataset, load_both_datasets

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)


def test_suppobox(X_test, y_test):
    X_test2, y_test2 = load_features_dataset(dataset="suppobox")
    X_test = np.concatenate((X_test, X_test2))
    y_test = np.concatenate((y_test, y_test2))
    return shuffle(X_test, y_test, random_state=42)


if __name__ == '__main__':
    X, y = load_both_datasets()
    test_split = 0.33
    # batch_size = 30
    epochs = 60
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    batch_size = 40
    # for batch_size in range(10, 110, 10):
    # model = Model(directory="saved_models/test_60/verysmall_35")
    # model = Model(model=verysmall_baseline(), directory="test_%s/verysmall_%s" % (epochs, batch_size))
    # model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=test_split, early=False)
    # model.classification_report(X_test, y_test, plot=False)

    # model.plot_AUC(X_test, y_test)
    pass
