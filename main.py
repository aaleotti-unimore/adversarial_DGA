import sys
import logging

import time
from keras.layers import Dense
from keras.models import Sequential, model_from_json

sys.path.append("../detect_DGA")

from sklearn.model_selection import train_test_split
from model import Model, compare, cross_val
from features.data_generator import *
from detect_DGA import plot_classification_report

logger = logging.getLogger(__name__)


def large_baseline():
    model = Sequential()
    model.add(Dense(18, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def reduced_baseline():
    """
    Modello ridotto
    :return:
    """
    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def pierazzi_baseline():
    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # model = Model(directory="saved models/2017-09-15 12:41 kula")
    # model.load_results()

    ##
    X, y = load_both_datasets()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    model = Model(model=large_baseline())
    model.fit(X_train, y_train, validation_data=(X_test, y_test))
    logger.info("\n%s" % model.test_model(X_test, y_test, plot=True))
    time.sleep(5)

    model2 = Model(model=reduced_baseline())
    model2.fit(X_train, y_train, validation_data=(X_test, y_test))
    logger.info("\n%s" % model2.test_model(X_test, y_test, plot=True))
    time.sleep(5)

    model3 = Model(model=pierazzi_baseline())
    model3.fit(X_train, y_train, validation_data=(X_test, y_test))
    logger.info("\n%s" % model3.test_model(X_test, y_test, plot=True))

    pass
