import logging
import os
import random as rn
import sys
import pandas as pd
import tensorflow as tf
import keras as K
import matplotlib
import numpy as np
from sklearn.utils import shuffle

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
sys.path.append("../detect_DGA")

from sklearn.model_selection import train_test_split
from classifier_model import Model, pierazzi_baseline, pierazzi_normalized_baseline, verysmall_baseline, lstm_baseline
# from features.data_generator import load_features_dataset, load_both_datasets
from detect_DGA import MyClassifier

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


def test_lstm():
    legit = pd.DataFrame(
        pd.read_csv('dataset/xaa',
                    header=None,
                    index_col=False
                    ).sample(100)
    )
    lstm_generated = pd.DataFrame(
        pd.read_csv('saved_models/lstm/generated.txt',
                    header=None,
                    index_col=False)
    )
    y_generated = np.ravel(np.zeros(len(lstm_generated), dtype=int))
    y_legit = np.ravel(np.ones(len(legit), dtype=int))

    X = pd.concat((lstm_generated, legit), axis=0)
    y = np.concatenate((y_generated, y_legit))
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y = lb.fit_transform(y=y)
    from features.features_extractors import get_feature_union
    ft = get_feature_union(-1)
    X = ft.fit_transform(X)
    return X, y


def load_domains(n_samples=None):
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    df = pd.DataFrame(pd.read_csv("../detect_DGA/datasets/legit_dga_domains.csv", sep=","))
    if n_samples:
        df = df.sample(n=n_samples, random_state=42)
    X = df['domain'].values
    y = np.ravel(lb.fit_transform(df['class'].values))
    return X, y


def LSTM_generator_dataset(path="dataset/xaa"):
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 15
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return X, y, maxlen, chars

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == '__main__':
    X, y, maxlen, chars = LSTM_generator_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train)
    model = Model(model=lstm_baseline(maxlen, chars), directory="lstm_model")
    model.fit(X_train, y_train, stdscaler=False, batch_size=32, epochs=128, validation_split=0.33, early=False)

    diversity = 1.2
    print('----- diversity:', diversity)
    generated = ''

    text = open("dataset/xaa").read().lower()
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    import random
    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: ' + sentence)
    print(generated)
    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))

        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        print(x)
        preds = model.model.predict_(x, verbose=0)[0]
        print(preds)
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        print(generated)
        sentence = sentence[1:] + next_char
        print(sentence)

        sys.stdout.write(next_char)
        sys.stdout.flush()

    print()

    # model.classification_report(X_test, y_test, plot=False)
    # X, y = test_lstm()

    # X, y = load_both_datasets()
    # test_split = 0.33
    # batch_size = 30
    # epochs = 60
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    # batch_size = 40
    # # for batch_size in range(10, 110, 10):
    # model = Model(directory="saved_models/test_60/pieraz_35_BEST")
    # model = Model(model=verysmall_baseline(), directory="test_%s/verysmall_%s" % (epochs, batch_size))
    # model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=test_split, early=False)
    # model.classification_report(X, y, plot=False)

    # model.plot_AUC(X_test, y_test)
    pass
