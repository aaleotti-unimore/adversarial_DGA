import string
import random
import pandas as pd
import numpy as np
from keras.models import Sequential


class Generator:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def generate_dataset():
    li = []
    for N in range(6, 10):
        for i in range(0, 100):
            li.append(string_generator(N))

    X = pd.DataFrame(li)
    y = np.chararray([len(li), 1], itemsize=3)
    y[:] = "dga"

    return X, y


def string_generator(N):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))
