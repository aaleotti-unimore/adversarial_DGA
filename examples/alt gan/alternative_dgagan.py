import argparse
import math
import logging
import string
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Input
from keras import backend as K
from keras.layers import Conv1D, Dropout, MaxPooling1D, concatenate, LSTM, RepeatVector, Dense, Embedding, Lambda, TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

CUDA_VISIBLE_DEVICES = 0
K.set_learning_phase(0)
print("set learning phase to %s" % K.learning_phase())


def generator_model(summary=None):
    dropout_value = 0.1
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    dec_convs = []
    lstm_vec_dim = 128
    timesteps = 15
    word_index = 38
    batch_size = 256

    # In: (batch_size, 128),
    # Out: (batch_size, timesteps, word_index)
    dec_inputs = Input(shape=(lstm_vec_dim,),
                       name="Generator_Input")
    # Repeating input by "timesteps" times. expected output (batch_size, 128, 15)
    decoded = RepeatVector(timesteps, name="gen_repeate_vec")(dec_inputs)
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
    # decoded = Dense(self.word_index, activation='sigmoid', name="gen_dense")(decoded)
    decoded = TimeDistributed(Dense(word_index, activation='softmax'), name='decoder_end')(
        decoded)  # output_shape = (samples, maxlen, max_features )

    G = Model(inputs=dec_inputs, outputs=decoded, name='Generator')
    if summary:
        G.summary()
    plot_model(G, to_file="images/generator.png", show_shapes=True)
    return G


def discriminator_model(summary=None):
    dropout_value = 0.1
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    enc_convs = []
    embedding_vec = 20  # lunghezza embedding layer
    timesteps = 15
    word_index = 38
    lstm_vec_dim = 128
    # In: (batch_size, timesteps),
    # Out: (batch_size, 128)

    # noise = K.random_uniform(shape=(256, timesteps,), maxval=1, minval=0, dtype='float32', seed=42)
    # print("noise : %s" % K.print_tensor(noise))
    discr_inputs = Input(shape=(timesteps,),
                         # tensor=noise,
                         name="Discriminator_Input")
    # print('enc_inputs: %s' % K.print_tensor(discr_inputs))
    # embedding layer. expected output ( batch_size, timesteps, embedding_vec)
    discr = Embedding(word_index, embedding_vec, input_length=timesteps)(discr_inputs)
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
    discr = LSTM(lstm_vec_dim)(discr)
    discr = Dense(1, activation='sigmoid')(discr)
    D = Model(inputs=discr_inputs, outputs=discr, name='Discriminator')
    if summary:
        D.summary()
    plot_model(D, to_file="images/discriminator.png", show_shapes=True)
    return D


def sampling(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # preds = K.expand_dims(preds,axis=0)
    # print(preds)
    preds = K.log(preds) / temperature
    exp_preds = K.exp(preds)
    preds = exp_preds / K.sum(exp_preds)
    return K.argmax(preds)


def generator_containing_discriminator(g, d, timesteps=15):
    model = Sequential()
    model.add(g)
    model.add(Lambda(lambda x: sampling(x), output_shape=(timesteps,), name="Sampling"))
    d.trainable = False
    model.add(d)
    return model


# def combine_images(generated_images):
#     num = generated_images.shape[0]
#     width = int(math.sqrt(num))
#     height = int(math.ceil(float(num) / width))
#     shape = generated_images.shape[1:3]
#     image = np.zeros((height * shape[0], width * shape[1]),
#                      dtype=generated_images.dtype)
#     for index, img in enumerate(generated_images):
#         i = int(index / width)
#         j = index % width
#         image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
#             img[:, :, 0]
#     return image


def train(BATCH_SIZE):
    # caricamento
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # X_train = X_train[:, :, :, None]
    # X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    maxlen = 15
    X_train, word_index = build_dataset(n_samples=BATCH_SIZE, maxlen=maxlen)

    # compulazione modelli
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d, timesteps=maxlen)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")  # compiling generator
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    # training
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 128))

            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            print("image_batch size")
            print(image_batch.shape)
            print(image_batch)

            generated_images = g.predict(noise, verbose=1)
            print("generated images shape:")
            print(generated_images.shape)
            generated_images = noise_sampling(generated_images)
            print("sampled generated images shape")
            print(generated_images.shape)

            # if index % 20 == 0:
            #     image = combine_images(generated_images)
            #     image = image * 127.5 + 127.5
            #     Image.fromarray(image.astype(np.uint8)).save(
            #         str(epoch) + "_" + str(index) + ".png")

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1.0, 1.0, (BATCH_SIZE, 128))

            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def build_dataset(n_samples=None, maxlen=15):
    # fix random seed for reproducibility
    np.random.seed(7)
    path = "/home/archeffect/PycharmProjects/adversarial_DGA/dataset/legitdomains.txt"
    n_samples = 10000

    # loading db
    # lb = LabelBinarizer()
    df = pd.DataFrame(pd.read_csv("/home/archeffect/PycharmProjects/adversarial_DGA/dataset/legitdomains.txt",
                                  sep=" ",
                                  header=None,
                                  names=['domain'])
                      )

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
    print("X shape " + str(X.shape))
    return X, len(tk.word_index)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
