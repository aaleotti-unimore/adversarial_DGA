from __future__ import print_function
import json
import logging
import os
import random as rn
import socket
import sys
import time
from datetime import datetime
from tempfile import mkdtemp

from matplotlib import pyplot as plt

sys.path.append("../detect_DGA")

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ProgbarLogger, Callback
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from numpy.random import RandomState
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

from plot_module import plot_classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)

tf.set_random_seed(1234)


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """

    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "Epoch: %i, %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.iteritems()))
        self.print_fcn(msg)


class Model:
    def __init__(self, model=None, directory=None):
        self.model = model

        self.directory = os.path.join("saved models", directory)
        if not os.path.exists(self.directory):
            # crea la cartella
            os.makedirs(self.directory)

        self.init_logger()
        if not self.model:
            self.model = self.__load_model()

    def init_logger(self):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger(__name__)
        hdlr = logging.FileHandler(os.path.join(self.directory, 'results.log'))
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    #
    # def __call__(self, *args, **kwargs):
    #     return self.model
    #
    # def __make_exp_dir(self, directory):
    #     if not os.path.exists(directory):
    #         directory = os.path.join("saved models", time.strftime("%c"))
    #         os.makedirs(directory)
    #
    #     if socket.gethostname() == "classificatoredga":
    #         directory = "kula_" + directory
    #     return directory

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
        try:
            model = model_from_json(open(os.path.join(self.directory, 'model_architecture.json')).read())
            model.load_weights(os.path.join(self.directory, 'model_weights.h5'))
            model.compile(loss='binary_crossentropy', optimizer='adam')
            return model
        except IOError as e:
            self.logger.error(e)

    def print_results(self, results, to_console=False):
        for key, value in sorted(results.iteritems()):
            if not "time" in key:
                foo = "%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100)
                if to_console:
                    print(foo)
                else:
                    self.logger.info(foo)
            else:
                foo = "%s: %.2fs (%.2f)s" % (key, value.mean(), value.std())
                if to_console:
                    print(foo)
                else:
                    self.logger.info(foo)

    def save_results(self, results):
        self.print_results(results)
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
        self.print_results(results, to_console=True)
        return results

    def get_model(self):
        return self.model

    def get_directory(self):
        return self.directory

    def classification_report(self, X, y, plot=True, save=True):
        std = StandardScaler()
        std.fit(X=X)
        X = std.transform(X=X)
        pred = self.model.predict(X)
        y_pred = [round(x) for x in pred]

        report = classification_report(y_pred=y_pred, y_true=y, target_names=['DGA', 'Legit'])
        if save:
            self.logger.info("\n%s" % report)
            if plot:
                plot_classification_report(classification_report=report,
                                           directory=self.directory)
        else:
            print(report)

    def fit(self, X, y, validation_data, batch_size=5, epochs=100, verbose=2):
        dirtemp = os.path.join(self.directory, "tensorboard")
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto')
        tensorboard = TensorBoard(log_dir=dirtemp,
                                  write_graph=False,
                                  write_images=False,
                                  histogram_freq=0)


        std = StandardScaler()
        X = std.fit_transform(X=X)

        self.model.fit(X, y, batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[early,
                                  tensorboard,
                                  LoggingCallback(print_fcn=self.logger.info)],
                       validation_data=validation_data,
                       verbose=verbose,
                       )
        self.save_model()

    def plot_AUC(self, X_test, y_test, save=True):
        std = StandardScaler()
        X_test = std.fit_transform(X_test)
        y_score = self.model.predict_proba(X_test)

        # y_score = [round(x) for x in y_score]
        # round_ = np.vectorize(round)
        # y_score = round_(y_score)

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
        if save:
            dirplt = os.path.join(self.directory, 'roc_plot.png')
            plt.savefig(dirplt, format="png")
        else:
            plt.show()

    def cross_val(self, X, y, save=False):
        t0 = datetime.now()
        self.logger.info("Starting cross validation at %s" % t0)

        _cachedir = mkdtemp()
        _memory = joblib.Memory(cachedir=_cachedir, verbose=2)
        pipeline = Pipeline(
            [('standardize', StandardScaler()),
             ('mlp', KerasClassifier(build_fn=pierazzi_baseline,
                                     epochs=100,
                                     batch_size=5,
                                     verbose=2))],
            memory=_memory)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RandomState())

        results = cross_validate(pipeline, X, y, cv=kfold, n_jobs=-1, verbose=2,
                                 scoring=['precision', 'recall', 'f1', 'roc_auc'])

        self.logger.info("Cross Validation Ended. Elapsed time: %s" % (datetime.now() - t0))
        if save:
            time.sleep(2)
            model = Model(pipeline.named_steps['mlp'].build_fn())
            model.get_model().summary(print_fn=self.logger.info)

            model.save_results(results)
            model.save_model()

            return model


def large_baseline():
    model = Sequential()

    model.add(Dense(18, input_dim=9, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])

    return model


def reduced_baseline():
    """
    Modello ridotto
    :return:
    """
    model = Sequential()

    model.add(Dense(9, input_dim=9, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(4, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])

    return model


def pierazzi_baseline(weights_path=None):
    model = Sequential()

    model.add(Dense(9, input_dim=9, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(64, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer='normal', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mae'])

    return model
