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


# def test_suppobox(X_test, y_test):
#     X_test2, y_test2 = load_features_dataset(dataset="suppobox")
#     X_test = np.concatenate((X_test, X_test2))
#     y_test = np.concatenate((y_test, y_test2))
#     return shuffle(X_test, y_test, random_state=42)


def both_datasets():
    legit = pd.DataFrame(
        pd.read_csv('/home/archeffect/PycharmProjects/adversarial_DGA/resources/datasets/all_legit.txt',
                    header=None,
                    index_col=False
                    ).sample(5000)
    )
    y_legit = np.ravel(np.ones(len(legit), dtype=int))

    generated = pd.read_csv(
        "/home/archeffect/PycharmProjects/adversarial_DGA/autoencoder_experiments/20171218-101804/samples.txt",
        index_col=None, header=None).sample(5000)
    y_generated = np.ravel(np.zeros(len(generated), dtype=int))

    X = pd.concat((generated, legit), axis=0)
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


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


if __name__ == '__main__':
    # model.classification_report(X_test, y_test, plot=False)
    X, y = both_datasets()
    # X = X.sample(n=1000, random_state=42)
    # print(X)

    # test_split = 0.33
    # batch_size = 30
    # epochs = 60
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    # batch_size = 40
    # # for batch_size in range(10, 110, 10):
    model = Model(
        directory="/home/archeffect/PycharmProjects/adversarial_DGA/neuralnetwork_classifier/saved_models/pieraz_norm_30_100")
    model.classification_report(X=X, y=y, plot=False, save=False,directory="/home/archeffect/PycharmProjects/adversarial_DGA/autoencoder_experiments/20171218-101804")
    # # model = Model(model=verysmall_baseline(), directory="test_%s/verysmall_%s" % (epochs, batch_size))
    # # model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=test_split, early=False)
    # # model.classification_report(X, y, plot=False)
    # print(model.get_model().predict(['ronncacncoouctm']))
    # model.plot_AUC(X_test, y_test)
    pass
