import sys
import logging

sys.path.append("../detect_DGA")

from sklearn.model_selection import train_test_split
from model import Model, compare, cross_val
from features.data_generator import *

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # model = Model(directory="saved models/2017-09-15 12:41 kula")
    # model.load_results()
    X, y = load_both_datasets()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = cross_val(X_train,y_train)
    model.fit(X_train, y_train)
    print(model.test_model(X_test, y_test))
    pass
