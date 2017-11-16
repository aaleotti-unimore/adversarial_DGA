import string

import numpy as np
import pandas as pd
from keras import Input
from keras import backend as K
from keras.layers import Conv1D, Dropout, MaxPooling1D, concatenate, LSTM, RepeatVector, Dense, Lambda, Embedding
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model


def generate_dataset(n_samples=None, maxlen=15):
    df = pd.DataFrame(
        pd.read_csv("/home/archeffect/PycharmProjects/adversarial_DGA/dataset/legitdomains.txt", sep=" ", header=None,
                    names=['domain']))
    if n_samples:
        df = df.sample(n=n_samples, random_state=42)
    X_ = df['domain'].values
    # y = np.ravel(lb.fit_transform(df['class'].values))

    # preprocessing text
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
    inv_map = {v: k for k, v in tk.word_index.iteritems()}

    return X, len(tk.word_index), inv_map


class GAN_Model(object):
    def __init__(self, batch_size, timesteps, word_index):
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.word_index = word_index
        self.lstm_vec_dim = 128

        self.D = None  # discriminator
        self.G = None  # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W-F+2P)/S+1
    def discriminator(self, summary=None):
        if self.D:
            return self.D

        dropout_value = 0.1
        cnn_filters = [20, 10]
        cnn_kernels = [2, 3]
        enc_convs = []
        embedding_vec = 20  # lunghezza embedding layer

        # In: (batch_size, timesteps),
        # Out: (batch_size, 128)


        discr_inputs = Input(shape=(self.timesteps,),
                             name="Discriminator_Input")

        # embedding layer. expected output ( batch_size, timesteps, embedding_vec)
        discr = Embedding(self.word_index, embedding_vec, input_length=self.timesteps)(discr_inputs)
        for i in range(2):
            conv = Conv1D(cnn_filters[i],
                          cnn_kernels[i],
                          padding='same',
                          activation='relu',
                          strides=1,
                          name='discr_conv%s' % i)(discr)

            conv = Dropout(dropout_value, name='discr_dropout%s' % i)(conv)
            conv = MaxPooling1D()(conv)
            enc_convs.append(conv)

        # concatenating CNNs. expected output (batch_size, 7, 30)
        discr = concatenate(enc_convs)
        # LSTM. expected out (batch_size, 128)
        discr = LSTM(self.lstm_vec_dim)(discr)
        discr = Dense(1, activation='sigmoid')(discr)

        self.D = Model(inputs=discr_inputs, outputs=discr, name='Discriminator')
        if summary:
            self.D.summary()
        plot_model(self.D, to_file="discriminator.png", show_shapes=True)
        return self.D

    def generator(self, summary=None):

        if self.G:
            return self.G

        dropout_value = 0.1
        cnn_filters = [20, 10]
        cnn_kernels = [2, 3]
        dec_convs = []

        # In: (batch_size, 128),
        # Out: (batch_size, timesteps, word_index)
        dec_inputs = Input(shape=(self.lstm_vec_dim,),
                           name="Generator_Input")
        # Repeating input by "timesteps" times. expected output (batch_size, 128, 15)
        decoded = RepeatVector(self.timesteps, name="gen_repeate_vec")(dec_inputs)
        decoded = LSTM(self.lstm_vec_dim, return_sequences=True, name="gen_LSTM")(decoded)

        for i in range(2):
            conv = Conv1D(cnn_filters[i],
                          cnn_kernels[i],
                          padding='same',
                          activation='relu',
                          strides=1,
                          name='gen_conv%s' % i)(decoded)
            conv = Dropout(dropout_value, name="gen_dropout%s" % i)(conv)
            dec_convs.append(conv)

        decoded = concatenate(dec_convs)
        decoded = Dense(self.word_index, activation='sigmoid', name="gen_dense")(decoded)

        self.G = Model(inputs=dec_inputs, outputs=decoded, name='Generator')
        if summary:
            self.G.summary()
        plot_model(self.G, to_file="generator.png", show_shapes=True)
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(Lambda(lambda x: K.mean(x,axis=1), output_shape=(self.timesteps,), name="Sampling"))
        self.AM.get_layer(name='Sampling').trainable = False
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy']
                        )
        self.AM.summary()
        plot_model(self.AM, to_file="adversial.png", show_shapes=True)
        return self.AM

    def sampling(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = K.log(preds) / temperature
        exp_preds = K.exp(preds)
        preds = exp_preds / K.sum(exp_preds)
        return K.argmax(preds, axis=2)


def noise_sampling(preds):
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
    maxlen = 15
    nsamples = 10
    X, word_index, inv_map = generate_dataset(n_samples=nsamples, maxlen=maxlen)
    adv = GAN_Model(nsamples, maxlen, word_index=word_index)
    # print(X)
    # noise = np.random.uniform(0,1,size=[nsamples,128])
    noise = np.ones(shape=[nsamples, 128])
    y = np.ones(shape=[nsamples, 1])

    adv.generator().compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                        )
    pred = adv.adversarial_model().train_on_batch(x=noise,y=y)
    print(pred)
