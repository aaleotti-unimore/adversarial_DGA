import sys
import logging
import numpy as np
import pandas as pd
import random, string
import pprint

sys.path.append('/home/archeffect/PycharmProjects/detect_DGA/')
import detect_DGA
from features import data_generator
from features.features_extractors import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)

pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(precision=3, suppress=True)

lb = LabelBinarizer()

logger = logging.getLogger(__name__)

n_samples = 20
n_jobs_pipeline = 2

ft = FeatureUnion(
    transformer_list=[
        ('mcr', MCRExtractor()),
        ('ns1', NormalityScoreExtractor(1)),
        ('ns2', NormalityScoreExtractor(2)),
        ('ns3', NormalityScoreExtractor(3)),
        ('ns4', NormalityScoreExtractor(4)),
        ('ns5', NormalityScoreExtractor(5)),
        ('len', DomainNameLength()),
        ('vcr', VowelConsonantRatio()),
        ('ncr', NumCharRatio()),
    ],
    n_jobs=n_jobs_pipeline
)


def load_legit_dataset():
    logger.info("samples %s" % n_samples)
    path = 'dataset/all_legit.txt'
    df = pd.read_csv(path, sep=' ', header=None, names=['domain', 'type'], usecols=['domain']).sample(n_samples,
                                                                                                      random_state=np.random.RandomState())
    df['type'] = 1
    X = DomainExtractor().transform(df['domain'].values.reshape(-1, 1))
    y = np.ravel(df['type'].values)
    return X, y


def prova(X, y):
    features = ft.transform(X)
    y_pred = detect_DGA.detect(X)
    df = pd.DataFrame(
        np.c_[X, features, y, y_pred],
        columns=['domain', 'mcr', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'len', 'vcr', 'ncr', 'label', 'prediction'])

    df['mcr'] = df['mcr'].apply(pd.to_numeric)
    df['ns1'] = df['ns1'].apply(pd.to_numeric)
    df['ns2'] = df['ns2'].apply(pd.to_numeric)
    df['ns3'] = df['ns3'].apply(pd.to_numeric)
    df['ns4'] = df['ns4'].apply(pd.to_numeric)
    df['ns5'] = df['ns5'].apply(pd.to_numeric)
    df['ncr'] = df['ncr'].apply(pd.to_numeric)
    df['vcr'] = df['vcr'].apply(pd.to_numeric)
    df['len'] = df['len'].apply(pd.to_numeric)

    print(df)
    print("")
    print("MEAN")
    print(pd.DataFrame(np.mean(features, axis=0).reshape(1, 9),
                       columns=['mcr', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'len', 'vcr', 'ncr']))
    print("")
    print(classification_report(y_true=y, y_pred=y_pred))


def __random_line(afile):
    import random
    lines = open(afile).read().splitlines()
    return random.choice(lines)


def test_sup():
    engdict = "dataset/suppodict.txt"

    N = 1000
    li = []
    for i in range(0, N):
        w1 = __random_line(engdict)
        w1 += __random_line(engdict)
        li.append(w1)

    X = pd.DataFrame(li)
    y = np.zeros(len(X), dtype=int)
    prova(X, y)


if __name__ == '__main__':
    # prova()
    test_sup()
