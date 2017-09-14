import json
import logging
import os
import socket
import time
from datetime import datetime
from tempfile import mkdtemp

import random as rn
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

import numpy as np
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from numpy.random.mtrand import RandomState
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.data_generator import load_both_datasets

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


class Model:
    def __init__(self, directory=None):
        self.now = time.strftime("%Y-%m-%d %H:%M")
        if directory:
            self.directory = directory
            self.model = self.__load_model()
        else:
            self.directory = self.__make_exp_dir()
            self.model = self.__create_baseline()

        self.logger = logging.getLogger(__name__)
        hdlr = logging.FileHandler(os.path.join(self.directory, 'results.log'))
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

    def __create_baseline(self):
        """ baseline model
        """
        # create model
        model = Sequential()
        model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64, kernel_initializer='normal', activation='relu'))
        # model.add(Dense(9, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.summary()

        # TODO cercare esempi di features float binarizzate

        return model

    def cross_val(self, n_samples=None):
        t0 = datetime.now()
        self.logger.info("Starting cross validation at %s" % t0)

        _cachedir = mkdtemp()
        _memory = joblib.Memory(cachedir=_cachedir, verbose=0)
        X, y = load_both_datasets(n_samples,verbose=True)
        # X, y = load_features_dataset(n_samples)
        estimators = [('standardize', StandardScaler()),
                      ('mlp', KerasClassifier(build_fn=self.__create_baseline, epochs=100, batch_size=5, verbose=0))]

        pipeline = Pipeline(estimators, memory=_memory)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RandomState())

        results = cross_validate(pipeline, X, y, cv=kfold, n_jobs=-1, verbose=1,
                                 scoring=['precision', 'recall', 'f1', 'roc_auc'])
        time.sleep(2)
        for key, value in results.iteritems():
            if not "time" in key:
                self.logger.info("%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100))
            else:
                self.logger.info("%s: %.2fs (%.2f)s" % (key, value.mean(), value.std()))

        model = pipeline.named_steps['mlp'].build_fn()
        model.summary(print_fn=self.logger.info)

        self.__save_model(results)

        self.logger.info("Cross Validation Ended. Elapsed time: %s" % (datetime.now() - t0))

    def __save_model(self, results=None):
        # saving model
        json_model = self.model.to_json()
        if results:
            _res = {k: v.tolist() for k, v in results.items()}
            with open(os.path.join(self.directory, 'data.json'), 'w') as fp:
                try:
                    json.dump(_res, fp, sort_keys=True, indent=4)
                except BaseException as e:
                    self.logger.error(e)

        open(os.path.join(self.directory, 'model_architecture.json'), 'w').write(json_model)
        self.logger.info("model saved to model_architecture.json")
        # saving weights
        self.model.save_weights(os.path.join(self.directory, 'model_weights.h5'), overwrite=True)
        self.logger.info("model weights saved to model_weights.h5")
        plot_model(self.model, to_file=os.path.join(self.directory, "model.png"), show_layer_names=True,
                   show_shapes=True)
        self.logger.info("network diagram plotted to model.png")

    def __load_model(self):
        # loading model
        model = model_from_json(open(os.path.join(self.directory, 'model_architecture.json')).read())
        model.load_weights(os.path.join(self.directory, 'model_weights.h5'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def __make_exp_dir(self):
        directory = "saved models/" + self.now
        if socket.gethostname() == "classificatoredga":
            directory += " kula"
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    def test_model(self, X, y):
        pred = self.model.predict(X)
        y_pred = [round(x) for x in pred]
        return classification_report(y_pred=y_pred, y_true=y, target_names=['DGA', 'Legit'])

    def model_stats(self):
        with open(os.path.join(self.directory, "data.json"), 'rb') as fd:
            d = json.load(fd)

        d = {k: np.asfarray(v) for k, v in d.iteritems()}
        for key, value in sorted(d.iteritems()):
            if not "time" in key:
                self.logger.info("%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100))
            else:
                self.logger.info("%s: %.2fs (%.2f)s" % (key, value.mean(), value.std()))

    def fit(self, X, y):
        std = StandardScaler()
        std.fit(X=X)
        X = std.transform(X=X)
        self.model.fit(X, y, verbose=0)
