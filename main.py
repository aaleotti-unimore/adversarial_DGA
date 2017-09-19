import sys
import logging

import numpy as np
from sklearn.utils import shuffle
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
sys.path.append("../detect_DGA")

from sklearn.model_selection import train_test_split
from model import Model, pierazzi_baseline, large_baseline, reduced_baseline
from features.data_generator import load_features_dataset, load_both_datasets

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_suppobox(X_test, y_test):
    X_test2, y_test2 = load_features_dataset(dataset="suppobox")
    X_test = np.concatenate((X_test, X_test2))
    y_test = np.concatenate((y_test, y_test2))
    return shuffle(X_test, y_test, random_state=42)


if __name__ == '__main__':
    # model = Model(directory="saved models/2017-09-15 12:41 kula")
    # model.load_results()
    X, y = load_both_datasets()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # X_test, y_test = test_suppobox(X_test, y_test)
    # X_test, y_test = test_suppobox(X_test,y_test)

    model = Model(model=large_baseline(), directory="large BatchNormalized")
    model.fit(X_train, y_train, validation_data=(X_test, y_test))
    model.classification_report(X_test, y_test, plot=False)
    # model.plot_AUC(X_test, y_test)

    model2 = Model(model=reduced_baseline(), directory="reduced BatchNormalized")
    model2.fit(X_train, y_train, validation_data=(X_test, y_test))
    model2.classification_report(X_test, y_test, plot=False)
    # model2.plot_AUC(X_test, y_test)

    # model3.cross_val(X_train, y_train,save=False)
    pass
