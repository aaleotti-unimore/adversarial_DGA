import sys

sys.path.append("../detect_DGA")

from model import Model, compare, cross_val

if __name__ == '__main__':
    model = cross_val(n_samples=100)
    # model = Model(directory="saved models/2017-09-14 17:27")
    compare(model)
    pass
