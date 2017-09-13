import socket

if socket.gethostname() == "classificatoredga":
    print("Hi! i'm on kula!")
    import sys

    sys.path.append("../detect_DGA")

import time
import random as rn
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from features.features_extractors import *
from features.features_testing import *

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
hdlr = logging.FileHandler('results.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model


def cross_val(n_samples=None):
    _cachedir = mkdtemp()
    _memory = joblib.Memory(cachedir=_cachedir, verbose=0)
    X, y = __load_both_datasets(n_samples)
    estimators = [('standardize', StandardScaler()),
                  ('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0))]
    pipeline = Pipeline(estimators, memory=_memory)
    logger.debug("Starting StratifiedKFold")
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RandomState())
    logger.debug("Endend StratifiedKFold")
    results = cross_val_score(pipeline, X, y, cv=kfold)
    logger.info("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    model = pipeline.named_steps['mlp'].build_fn()
    model.summary()
    plot_model(model, to_file="model.png", show_layer_names=True, show_shapes=True)
    logger.info("network diagram plotted to model.png")
    __save_model(model)


def __load_both_datasets(n_samples=None):
    X1, y1 = load_features_dataset()
    X2, y2 = load_features_dataset(
        dataset=os.path.join(basedir, "datas/suppobox_dataset.csv"))
    X = np.concatenate((X1, X2), axis=0).astype(float)
    y = np.concatenate((y1, y2), axis=0).astype(int)

    if n_samples:
        return shuffle(X, y, random_state=RandomState(), n_samples=n_samples)
    return shuffle(X, y, random_state=RandomState())


def __save_model(model):
    # saving model
    now = time.strftime("%Y-%m-%d %H:%M")
    directory = "saved models/" + now
    if not os.path.exists(directory):
        os.makedirs(directory)
    json_model = model.to_json()

    open(os.path.join(directory, 'model_architecture.json'), 'w').write(json_model)
    logger.info("model saved to model_architecture.json")
    # saving weights
    model.save_weights(os.path.join(directory, 'model_weights.h5'), overwrite=True)
    logger.info("model weights saved to model_weights.h5")


def load_model(directory):
    # loading model
    model = model_from_json(open(os.path.join(directory, 'model_architecture.json')).read())
    model.load_weights(os.path.join(directory, 'model_weights.h5'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def deploy():
    n_samples = None
    t0 = time.time()
    logger.info("Starting new training at %s. n of samples: %s" % (time.clock(), n_samples))
    cross_val(n_samples)
    logger.info("Elapsed time: %s s" % (time.time() - t0))


if __name__ == '__main__':
    # import pprint
    #
    # for i in range(0, 10):
    #     X, y = load_features_dataset(20)
    #
    #     std = StandardScaler().fit(X)
    #     X = std.transform(X)
    #     model = load_model("saved models/2017-09-13 14:32")
    #
    #     results = model.evaluate(X,y)
    #     print(results)
    #     # pprint.pprint(model.metrics_name)
    #     # logger.info("%s %s" % (model.metrics_names[0], results))
    deploy()
    pass
