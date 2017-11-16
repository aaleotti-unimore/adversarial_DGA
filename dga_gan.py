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

logging.basicConfig()


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
    def __init__(self, batch_size):
        self.logger = logging.getLogger(__name__)
        self.x_train, word_index = self.build_dataset(n_samples=None)

        self.DCGAN = GAN_Model(batch_size=batch_size, timesteps=self.x_train.shape[1], word_index=word_index)
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=2, save_interval=0):
        self.logger.setLevel(logging.DEBUG)
        noise_input = None
        if save_interval > 0:
            # noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
            pass
        for i in range(train_steps):
            if i>0:
                self.logger.setLevel(logging.INFO)
            self.logger.debug("train step: %s" % i)
            # loading training set
            domains_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=batch_size / 2), :]
            # generating random noise
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size / 2, 128])
            self.logger.debug("noise shape:")
            self.logger.debug(noise.shape)
            # predict fake domains
            self.logger.debug("generating domains_fake...")
            domains_fake = self.generator.predict(noise)  # fake domains
            self.logger.debug("sampling fake domains....")
            domains_fake = self.noise_sampling(domains_fake)
            self.logger.debug("real domains shape")
            self.logger.debug(domains_train.shape)
            self.logger.debug("fake domains shape")
            self.logger.debug(domains_fake.shape)
            # concatenating fake and train domains, labeled with 0 (real) and 1 (fake)
            x = np.concatenate((domains_train, domains_fake))
            y = np.ones([batch_size, 1])  # size 2x batch size of x
            y[batch_size / 2:, :] = 0
            import tensorflow as tf
            # x = tf.convert_to_tensor(x)
            self.logger.debug("X:")
            self.logger.debug(x.shape)
            self.logger.debug(x)
            self.logger.debug("y:")
            self.logger.debug(y.shape)
            # self.logger.debug(y)
            # training discriminator
            ####
            self.logger.debug("training discriminator")
            d_loss = self.discriminator.train_on_batch(x=x, y=y)
            # dataset for adversial model
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 128])  # random noise
            y = np.ones([batch_size, 1])
            self.logger.debug(noise.shape)
            self.logger.debug(y.shape)
            # training adversial model
            a_loss = self.adversarial.train_on_batch(x=noise, y=y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            self.logger.info(log_mesg)

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
        self.logger.debug("word index: %s" % len(tk.word_index))
        seq = tk.texts_to_sequences(X_)
        # for x, s in zip(X_, seq):
        #     print(x, s)
        # print("")
        X = sequence.pad_sequences(seq, maxlen=maxlen)
        self.logger.debug("X shape after padding: " + str(X.shape))
        # print(X)

        return X, len(tk.word_index)

    def noise_sampling(self, preds):
        def __sample(preds, temperature=1.0):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds).astype('float32')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        domains = []
        for j in range(preds.shape[0]):
            word = []
            for i in range(preds.shape[1]):
                word.append(__sample(preds[j][i]))
            domains.append(word)

        return np.array(domains)


if __name__ == '__main__':
    batch_size = 1000
    mnist_dcgan = DGA_GAN(batch_size=batch_size)
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10000, batch_size=batch_size, save_interval=500)
    timer.elapsed_time()
