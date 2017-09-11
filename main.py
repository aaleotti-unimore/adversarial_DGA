import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer

from features.features_extractors import *
from features.features_testing import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)

pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(precision=3, suppress=True)

lb = LabelBinarizer()

logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('results.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)


def neuralnetwork():
    X1, y1 = load_features_dataset()
    X2, y2 = load_features_dataset(
        dataset=os.path.join(basedir, "datas/suppobox_dataset.csv"))
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=RandomState())

    logger.debug("X: %s" % str(X.shape))
    logger.debug("y: %s" % str(y.shape))
    np.random.seed(42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=RandomState())

    model = Sequential()
    model.add(Dense(18, input_dim=9, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    logger.info("fitting...")
    model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
    logger.info("fitting completed")
    model.save("model.h5")
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    plot_model(model, to_file="model.png")


if __name__ == '__main__':
    neuralnetwork()
    pass
    # prova()
    # test_sup()
