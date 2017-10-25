import logging
import string
import time

import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

from gan_model import GAN_Model


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


class DGA_GAN(object):
    def __init__(self):
        K.set_learning_phase(0)
        self.logger = logging.getLogger(__name__)
        self.x_train, word_index = self.build_dataset(n_samples=None)

        self.DCGAN = GAN_Model(timesteps=self.x_train.shape[1], word_index=word_index)
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):

        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):

            # load training set
            domains_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size), :]
            # generating random noise
            noise = K.random_uniform(minval=0, maxval=1, shape=(batch_size, 128))  # random noise
            print("domains_train shape:")
            print(domains_train.shape)
            print("noise shape: %s " % noise.shape)
            print(noise.shape)
            # predict fake domains
            domains_fake = self.generator.predict(
                noise,
                batch_size=batch_size,
                steps=None,
                verbose=1
            )  # fake domains
            # print("domains_fake shape:")
            # print(domains_fake.shape)

            # sampling of fake domains
            d_sampl = []
            for j in range(domains_fake.shape[0]):
                word = []
                for i in range(domains_fake.shape[1]):
                    k = self.sample(domains_fake[j][i])
                    word.append(k)
                d_sampl.append(word)

            # concatenating fake and train domains, labeled with 0 (real) and 1 (fake)
            domains_fake = np.array(d_sampl)

            x = np.concatenate((domains_train, domains_fake))
            y = np.ones([2 * batch_size, 1])  # size 2x batch size of x
            y[batch_size:, :] = 0

            print("Discr input shape: %s" % self.discriminator.input_shape)
            print("domains_fake new shape:")
            print(domains_fake.shape)
            print("X shape:")
            print(x.shape)

            # training discriminator
            d_loss = self.discriminator.train_on_batch(x, y)

            # dataset for adversial model
            noise = np.random.normal(-0.02, 0.02, size=[batch_size, 128])  # random noise
            y = np.ones([batch_size, 1])

            # training adversial model
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
    mnist_dcgan = DGA_GAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
    timer.elapsed_time()
