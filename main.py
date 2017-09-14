import socket

if socket.gethostname() == "classificatoredga":
    print("Hi! i'm on kula!")
    import sys

    sys.path.append("../detect_DGA")

import json
import time, datetime
import random as rn
import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from detect_DGA import detect
from features.features_extractors import *
from features.features_testing import *
from features.data_generator import load_both_datasets, get_balance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)

pd.options.display.float_format = '{:.2f}'.format

lb = LabelBinarizer()

logger = logging.getLogger(__name__)

# if socket.gethostname() == "classificatoredga":
hdlr = logging.FileHandler('run.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(18, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # TODO cercare esempi di features float binarizzate

    return model


def cross_val(directory, n_samples=None):
    _cachedir = mkdtemp()
    _memory = joblib.Memory(cachedir=_cachedir, verbose=0)
    X, y = load_both_datasets(n_samples)
    # X, y = load_features_dataset(n_samples)
    estimators = [('standardize', StandardScaler()),
                  ('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0))]

    pipeline = Pipeline(estimators, memory=_memory)
    # pipeline.fit(X, y)
    logger.info("fitting done")
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RandomState())

    results = cross_validate(pipeline, X, y, cv=kfold, n_jobs=-1, verbose=1,
                             scoring=['precision', 'recall', 'f1', 'roc_auc'])
    time.sleep(2)
    for key, value in results.iteritems():
        if not "time" in key:
            logger.info("%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100))
        else:
            logger.info("%s: %.2fs (%.2f)s" % (key, value.mean(), value.std()))

    model = pipeline.named_steps['mlp'].build_fn()
    model.summary(print_fn=logger.info)

    __save_model(directory, model, results)


def __save_model(directory, model, results=None):
    # saving model
    json_model = model.to_json()
    if results:
        _res = {k: v.tolist() for k, v in results.items()}
        with open(os.path.join(directory, 'data.json'), 'w') as fp:
            try:
                json.dump(_res, fp, sort_keys=True, indent=4)
            except BaseException as e:
                logger.error(e)

    open(os.path.join(directory, 'model_architecture.json'), 'w').write(json_model)
    logger.info("model saved to model_architecture.json")
    # saving weights
    model.save_weights(os.path.join(directory, 'model_weights.h5'), overwrite=True)
    logger.info("model weights saved to model_weights.h5")
    plot_model(model, to_file=os.path.join(directory, "model.png"), show_layer_names=True, show_shapes=True)
    logger.info("network diagram plotted to model.png")


def load_model(directory):
    # loading model
    model = model_from_json(open(os.path.join(directory, 'model_architecture.json')).read())
    model.load_weights(os.path.join(directory, 'model_weights.h5'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def __make_exp_dir():
    now = time.strftime("%Y-%m-%d %H:%M")
    directory = "saved models/" + now
    if socket.gethostname() == "classificatoredga":
        directory += " kula"
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory, now


def deploy(n_samples=None):
    directory, now = __make_exp_dir()

    hdlr = logging.FileHandler(os.path.join(directory, 'results.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    t0 = time.time()
    logger.info("Starting new training at %s. n of samples: %s" % (now, n_samples))
    cross_val(directory=directory, n_samples=n_samples)
    logger.info("Elapsed time: %s s" % (time.time() - t0))


def test_model(directory, X, y):
    model = load_model(directory)
    std = StandardScaler()
    std.fit(X=X)
    X = std.transform(X=X)
    pred = model.predict(X)
    y_pred = [round(x) for x in pred]
    print(classification_report(y_pred=y_pred, y_true=y, target_names=['DGA', 'Legit']))


def compare():
    datasets = {
        "legit-dga dataset": load_features_dataset(),
        "suppobox": load_features_dataset(dataset="suppobox"),
        "legit-dga dataset + suppobox": load_both_datasets()
    }

    for key, (X, y) in datasets.iteritems():
        print("")
        print("%s" % key)
        print("Neural Network")
        test_model("saved models/2017-09-13 19:32", X, y)
        print("Random Forest")
        detect(X, y)


if __name__ == '__main__':
    deploy()
    pass
