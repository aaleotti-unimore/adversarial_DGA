import socket
if socket.gethostname() == "classificatoredga":
    print("Hi! i'm on kula!")
    import sys
    sys.path.append("../detect_DGA")

import random as rn

import pydot
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)

pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(precision=3, suppress=True)

lb = LabelBinarizer()

if socket.gethostname() == "classificatoredga":
    logger = logging.getLogger(__name__)
    hdlr = logging.FileHandler('results.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def cross_val(n_samples=None):
    X, y = __load_both_datasets(n_samples)
    estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RandomState())
    results = cross_val_score(estimator, X, y, cv=kfold)
    print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def __load_both_datasets(n_samples=None):
    X1, y1 = load_features_dataset()
    X2, y2 = load_features_dataset(
        dataset=os.path.join(basedir, "datas/suppobox_dataset.csv"))
    X = np.concatenate((X1, X2), axis=0).astype(float)
    y = np.concatenate((y1, y2), axis=0).astype(int)

    if n_samples:
        return shuffle(X, y, random_state=RandomState(), n_samples=n_samples)
    return shuffle(X, y, random_state=RandomState())


def neuralnetwork(n_samples=None):
    X, y = __load_both_datasets(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)

    logger.debug("X: %s" % str(X.shape))
    logger.debug("y: %s" % str(y.shape))

    model = Sequential()
    model.add(Dense(18, input_dim=9, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='accuracy', patience=2)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    logger.info("fitting...")
    model.summary()
    model.fit(X_train, y_train,
              epochs=150,
              batch_size=100,
              verbose=1,
              # callbacks=[early_stopping],
              validation_split=0.10
              )
    # print(history.history)
    logger.info("fitting completed")
    model.save("model.h5")
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    plot_model(model, to_file="model.png")


def save_model(model):
    # saving model
    json_model = model.to_json()
    open('model_architecture.json', 'w').write(json_model)
    # saving weights
    model.save_weights('model_weights.h5', overwrite=True)


def load_model():
    # loading model
    model = model_from_json(open('model_architecture.json').read())
    model.load_weights('model_weights.h5')
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


if __name__ == '__main__':
    cross_val()
    pass
    # prova()
    # test_sup()
