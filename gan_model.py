import numpy as np
from keras import backend as K
from keras import Input
from keras.layers import Conv1D, Dropout, MaxPooling1D, concatenate, LSTM, RepeatVector, Dense, Lambda, Embedding
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.utils import plot_model


class GAN_Model(object):
    def __init__(self, timesteps, word_index):
        K.set_learning_phase(0)
        self.timesteps = timesteps
        self.word_index = word_index
        self.lstm_vec_dim = 128

        self.D = None  # discriminator
        self.G = None  # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W-F+2P)/S+1
    def discriminator(self):
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

        self.D = Model(inputs=discr_inputs, outputs=discr, name='Discriminator')
        self.D.summary()
        plot_model(self.D, to_file="discriminator.png", show_shapes=True)
        return self.D

    def generator(self):
        if self.G:
            return self.G

        dropout_value = 0.1
        cnn_filters = [20, 10]
        cnn_kernels = [2, 3]
        dec_convs = []

        # In: (batch_size, 128),
        # Out: (batch_size, timesteps, word_index)
        noise = K.random_uniform(shape=(256, self.lstm_vec_dim))
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
        self.AM.add(Lambda(lambda x: self.sampling(x), output_shape=(self.timesteps,),name="Sampling"))
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])
        self.AM.summary()
        plot_model(self.AM, to_file="adversial.png", show_shapes=True)
        return self.AM

    def sampling(self, x):
        def __sample(preds, temperature=1.0):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds).astype('float32')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        K.set_learning_phase(0)
        preds = K.eval(x)
        # print(preds)
        domains = []
        for j in range(preds.shape[0]):
            word = []
            for i in range(preds.shape[1]):
                word.append(__sample(preds[j][i]))
            domains.append(word)

        return K.variable(domains, dtype='int32')
