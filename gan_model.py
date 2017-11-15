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
    df = pd.DataFrame(pd.read_csv("/home/archeffect/PycharmProjects/adversarial_DGA/dataset/legitdomains.txt", sep=" ", header=None, names=['domain']))
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
        K.set_learning_phase(0)
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
        K.set_learning_phase(1)

        if self.D:
            return self.D

        dropout_value = 0.1
        cnn_filters = [20, 10]
        cnn_kernels = [2, 3]
        enc_convs = []
        embedding_vec = 20  # lunghezza embedding layer

        # In: (batch_size, timesteps),
        # Out: (batch_size, 128)

        # noise = K.random_uniform(shape=(256, self.timesteps,), maxval=1, minval=0, dtype='float32', seed=42)
        # print("noise : %s" % K.print_tensor(noise))
        discr_inputs = Input(shape=(self.timesteps,),
                             # tensor=noise,
                             name="Discriminator_Input")
        # print('enc_inputs: %s' % K.print_tensor(discr_inputs))
        # embedding layer. expected output ( batch_size, timesteps, embedding_vec)
        discr = Embedding(self.word_index, embedding_vec, input_length=self.timesteps)(discr_inputs)
        # print("Embedding: %s" % K.print_tensor(discr_inputs))
        # print("embedding shape %s " % discr.shape)
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
        discr = Dense(1,activation='sigmoid')(discr)

        self.D = Model(inputs=discr_inputs, outputs=discr, name='Discriminator')
        if summary:
            self.D.summary()
        plot_model(self.D, to_file="discriminator.png", show_shapes=True)
        return self.D

    def generator(self, summary=None):
        K.set_learning_phase(1)

        if self.G:
            return self.G

        dropout_value = 0.1
        cnn_filters = [20, 10]
        cnn_kernels = [2, 3]
        dec_convs = []

        # In: (batch_size, 128),
        # Out: (batch_size, timesteps, word_index)
        # noise = K.random_uniform(shape=(256, self.lstm_vec_dim))
        dec_inputs = Input(shape=(self.lstm_vec_dim,),
                           # tensor=noise,
                           name="Generator_Input")
        # Repeating input by "timesteps" times. expected output (batch_size, 128, 15)
        decoded = RepeatVector(self.timesteps, name="gen_repeate_vec")(dec_inputs)
        decoded = LSTM(128, return_sequences=True, name="gen_LSTM")(decoded)

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
        # self.AM.add(Lambda(lambda x: self.sampling(x), output_shape=(self.timesteps,), name="Sampling"))
        self.AM.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(self.timesteps,), name="pseudosampling"))
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])
        self.AM.summary()
        plot_model(self.AM, to_file="adversial.png", show_shapes=True)
        return self.AM

    def sampling(self, x):
        def __sample(preds, temperature=1.0):
            # helper function to sample an index from a probability array
            # preds = K.expand_dims(preds,axis=0)
            # print(preds)
            preds = K.log(preds) / temperature
            exp_preds = K.exp(preds)
            preds = exp_preds / K.sum(exp_preds)
            probas = np.random.multinomial(1, K.eval(preds), 1)
            return K.expand_dims(K.variable(np.argmax(probas)))

        result = None
        for i in range(1):
            result = __sample(x[i, 0, :])
            for j in range(K.int_shape(x)[1]):
                if j == 0:
                    continue
                c = __sample(x[i, j, :])
                result = K.concatenate([result, c], axis=0)

        for i in range(self.batch_size):
            if i == 0:
                continue
            cane = __sample(x[i, 0, :])
            for j in range(K.int_shape(x)[1]):
                if j == 0:
                    continue
                c = __sample(x[i, j, :])
                cane = K.concatenate([cane, c], axis=0)
            result = K.concatenate([result, cane], axis=0)

        return result


if __name__ == '__main__':
    gan = GAN_Model(batch_size=256, timesteps=15, word_index=38)
    X, word_index, inv_map = generate_dataset(12)
    # disc = gan.discriminator().predict_on_batch(X)
    # print("DISCRIMINATED")
    # print(disc)
    # generated = gan.generator().predict_on_batch(disc)
    # print("GENERATED")
    # print(generated.shape)
    noise = np.random.uniform(size=(12,128))
    print(gan.adversarial_model().predict_on_batch(noise))