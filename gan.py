'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import random
import time
import string
import pandas as pd
import numpy as np
import time
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

from pandas import read_csv
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import LSTM, RepeatVector
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import RMSprop
from keras.models import Sequential

import matplotlib.pyplot as plt

import logging


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


class DCGAN(object):
    def __init__(self, timesteps, word_index):
        self.timesteps = timesteps
        self.word_index = word_index
        self.dim = 128

        self.D = None  # discriminator
        self.G = None  # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W-F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        dropout = 0.1
        filters = [20, 10]
        kernels = [2, 3]
        enc_convs = []
        d = 20  # lunghezza vettore embedded
        # In: (batch_size, timesteps),
        # Out: (batch_size, 128)
        enc_inputs = Input(shape=(self.timesteps,),name="Discriminator_Input")
        encoded = Embedding(self.word_index, d, input_length=self.timesteps)(enc_inputs)
        for i in range(2):
            conv = Conv1D(filters[i],
                          kernels[i],
                          padding='same',
                          activation='relu',
                          strides=1)(encoded)
            conv = Dropout(dropout)(conv)
            conv = MaxPooling1D()(conv)
            enc_convs.append(conv)

        encoded = concatenate(enc_convs)
        encoded = LSTM(128)(encoded)
        self.D = Model(inputs=enc_inputs, outputs=encoded, name='Discriminator')
        self.D.summary()
        plot_model(self.D, to_file="discriminator.png", show_shapes=True)

        return self.D

    def generator(self):
        if self.G:
            return self.G
        dropout = 0.1
        filters = [20, 10]
        kernels = [2, 3]
        dec_convs = []

        # In: (batch_size, 128),
        # Out: (batch_size, timesteps, word_index)
        dec_inputs = Input(shape=(self.dim,),name="Generator_Input")
        decoded = RepeatVector(self.timesteps)(dec_inputs)
        decoded = LSTM(128, return_sequences=True)(decoded)

        for i in range(2):
            conv = Conv1D(filters[i],
                          kernels[i],
                          padding='same',
                          activation='relu',
                          strides=1)(decoded)
            conv = Dropout(dropout)(conv)
            dec_convs.append(conv)

        decoded = concatenate(dec_convs)
        decoded = Dense(self.word_index, activation='sigmoid')(decoded)

        self.G = Model(inputs=dec_inputs, outputs=decoded, name='Generator')
        self.G.summary()
        plot_model(self.G, to_file="generator.png", show_shapes=True)
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, \
                        metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, \
                        metrics=['accuracy'])
        return self.AM

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float32')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


class MNIST_DCGAN(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.x_train, word_index = self.build_dataset(n_samples=None)

        self.DCGAN = DCGAN(timesteps=self.x_train.shape[1], word_index=word_index)
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):

        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            domains_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :]
            noise = np.random.normal(-0.02, 0.02, size=[batch_size, 128])  # random noise
            domains_fake = self.generator.predict(noise)  # fake domains
            x = np.concatenate((domains_train, domains_fake))  # legit and fake domains
            y = np.ones([2 * batch_size, 1])  # dimensione 2x batch size di 1
            y[batch_size:, :] = 0  # prima meta, images_train, a zero
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.normal(-0.02, 0.02, size=[batch_size, 128])  # random noise
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

    def build_dataset(self, n_samples=None):
        # fix random seed for reproducibility
        np.random.seed(7)
        path = "/home/archeffect/PycharmProjects/adversarial_DGA/dataset/legitdomains.txt"
        n_samples = 256

        # loading db
        lb = LabelBinarizer()
        df = pd.DataFrame(pd.read_csv("dataset/legitdomains.txt", sep=" ", header=None, names=['domain']))
        if n_samples:
            df = df.sample(n=n_samples, random_state=42)
        X_ = df['domain'].values
        # y = np.ravel(lb.fit_transform(df['class'].values))

        # preprocessing text
        maxlen = 15
        tk = Tokenizer(char_level=True)
        tk.fit_on_texts(string.lowercase + string.digits + '-' + '.')
        print("word index: %s" % len(tk.word_index))
        seq = tk.texts_to_sequences(X_)
        # for x, s in zip(X_, seq):
        #     print(x, s)
        # print("")
        X = sequence.pad_sequences(seq, maxlen=maxlen)
        print("X shape after padding: " + str(X.shape))
        # print(X)

        return X, len(tk.word_index)


if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
    timer.elapsed_time()
