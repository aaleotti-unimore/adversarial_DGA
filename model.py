import json
import logging
import os
import random as rn
import socket
import sys
import time
from datetime import datetime
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from numpy.random import RandomState
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

from plot_module import plot_classification_report

sys.path.append("../detect_DGA")

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)

tf.set_random_seed(1234)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)

pd.options.display.float_format = '{:.2f}'.format

lb = LabelBinarizer()

logger = logging.getLogger(__name__)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')


class Model:
    def __init__(self, model=None, directory=None):
        self.now = time.strftime("%Y-%m-%d %H:%M")
        self.model = model
        if directory:
            self.directory = directory
            self.model = self.__load_model()
        else:
            self.directory = self.__make_exp_dir()

        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger(__name__)
        self.hdlr = logging.FileHandler(os.path.join(self.directory, 'results.log'))
        self.hdlr.setFormatter(self.formatter)
        self.logger.addHandler(self.hdlr)

    def __make_exp_dir(self):
        directory = "saved models/" + self.now
        if socket.gethostname() == "classificatoredga":
            directory += " kula"
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    def save_model(self):
        # saving model
        json_model = self.model.to_json()
        dirmod = os.path.join(self.directory, 'model_architecture.json')
        open(dirmod, 'w').write(json_model)
        self.logger.info("model saved to %s" % dirmod)
        # saving weights
        dirwe = os.path.join(self.directory, 'model_weights.h5')
        self.model.save_weights(dirwe, overwrite=True)
        self.logger.info("model weights saved to %s" % dirwe)
        dirplo = os.path.join(self.directory, "model.png")
        plot_model(self.model, to_file=dirplo, show_layer_names=True,
                   show_shapes=True)
        self.logger.info("network diagram saved to %s " % dirplo)

    def __load_model(self):
        # loading model
        model = model_from_json(open(os.path.join(self.directory, 'model_architecture.json')).read())
        model.load_weights(os.path.join(self.directory, 'model_weights.h5'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def save_results(self, results):
        for key, value in sorted(results.iteritems()):
            if not "time" in key:
                logger.info("%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100))
            else:
                logger.info("%s: %.2fs (%.2f)s" % (key, value.mean(), value.std()))

        _res = {k: v.tolist() for k, v in results.items()}
        with open(os.path.join(self.directory, 'data.json'), 'w') as fp:
            try:
                json.dump(_res, fp, sort_keys=True, indent=4)
            except BaseException as e:
                self.logger.error(e)

    def load_results(self):
        with open(os.path.join(self.directory, "data.json"), 'rb') as fd:
            results = json.load(fd)

        results = {k: np.asfarray(v) for k, v in results.iteritems()}
        for key, value in sorted(results.iteritems()):
            if not "time" in key:
                self.logger.info("%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100))
            else:
                self.logger.info("%s: %.2fs (%.2f)s" % (key, value.mean(), value.std()))
        return results

    def get_model(self):
        return self.model

    def get_directory(self):
        return self.directory

    def classification_report(self, X, y, plot=False):
        std = StandardScaler()
        std.fit(X=X)
        X = std.transform(X=X)
        pred = self.model.predict(X)
        y_pred = [round(x) for x in pred]

        repo = classification_report(y_pred=y_pred, y_true=y, target_names=['DGA', 'Legit'])
        if plot:
            plot_classification_report(classification_report=repo,
                                       directory=self.directory)
        return repo

    def fit(self, X, y, validation_data, batch_size=5, epochs=100, verbose=0):
        std = StandardScaler()
        std.fit(X=X)
        X = std.transform(X=X)
        self.model.fit(X, y, batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[early],
                       validation_data=validation_data,
                       verbose=verbose)
        self.save_model()

    def plot_AUC(self, X_test, y_test):
        std = StandardScaler()
        std.fit(X_test)
        X_test = std.transform(X_test)
        y_score = self.model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_score[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc, lw=1.5,
                 alpha=.8)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
                 label='Luck', alpha=.8)
        plt.xlim([0, 1.00])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        dirplt = os.path.join(self.directory, 'roc_plot.png')
        plt.savefig(dirplt, format="png")

    def cross_validate(self, X_train, y_train, scoring=None):
        if scoring is None:
            scoring = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc']

        std = StandardScaler()
        std.fit(X=X_train)
        X_train = std.transform(X=X_train)
        kc = KerasClassifier(build_fn=self.model, epochs=100, batch_size=5, verbose=0)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RandomState())
        self.save_results(cross_validate(kc, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1, verbose=1))


def create_baseline():
    """ baseline model
    """
    model = Sequential()
    model.add(Dense(18, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # TODO cercare esempi di features float binarizzate

    return model


def cross_val(X, y, model=None):
    if not model:
        model = create_baseline
    t0 = datetime.now()
    logger.info("Starting cross validation at %s" % t0)

    _cachedir = mkdtemp()
    _memory = joblib.Memory(cachedir=_cachedir, verbose=0)

    estimators = [('standardize', StandardScaler()),
                  ('mlp', KerasClassifier(build_fn=model, epochs=100, batch_size=5, verbose=0))]

    pipeline = Pipeline(estimators, memory=_memory)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RandomState())

    results = cross_validate(pipeline, X, y, cv=kfold, n_jobs=-1, verbose=1,
                             scoring=['precision', 'recall', 'f1', 'roc_auc'])
    time.sleep(2)
    model = Model(pipeline.named_steps['mlp'].build_fn())
    model.get_model().summary(print_fn=logger.info)

    model.save_results(results)
    model.save_model()

    logger.info("Cross Validation Ended. Elapsed time: %s" % (datetime.now() - t0))
    return model
