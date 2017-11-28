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
from keras.layers import Conv1D, Dropout, MaxPooling1D, concatenate, LSTM, RepeatVector, Dense, Embedding, Lambda, \
    TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.utils import plot_model, to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import pandas as pd
from sklearn.preprocessing import LabelBinarizer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

    discr_inputs = Input(shape=(timesteps, word_index),
                         name="Discriminator_Input")
    # embedding layer. expected output ( batch_size, timesteps, embedding_vec)
    manual_embedding = Dense(embedding_vec, activation='linear')
    discr = TimeDistributed(manual_embedding, name='manual_embedding', trainable=False)(
        discr_inputs)
    # discr = Embedding(word_index, embedding_vec, input_length=timesteps)(discr_inputs)
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
    discr = LSTM(lstm_vec_dim)(discr)
    discr = Dense(1, activation='sigmoid', kernel_initializer='normal')(discr)

    D = Model(inputs=discr_inputs, outputs=discr, name='Discriminator')
    if summary:
        D.summary()
    plot_model(D, to_file="images/discriminator.png", show_shapes=True)
    return D


def sampling(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = K.log(preds) / temperature
    exp_preds = K.exp(preds)
    preds = exp_preds / K.sum(exp_preds)
    return K.argmax(preds)


def generator_containing_discriminator(g, d, timesteps=15):
    model = Sequential()
    model.add(g)
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


def train(BATCH_SIZE, load_weights=True):
    # caricamento
    maxlen = 15
    n_samples = 50000
    X_train, word_index, inv_map = build_dataset(maxlen=maxlen, n_samples=n_samples)
    logger.debug(X_train.shape)
    # compilazione modelli
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d, timesteps=maxlen)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")  # compiling generator
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    tb_gen = TensorBoard(log_dir='.logs/gen', write_graph=False)
    tb_gen.set_model(g)
    tb_disc = TensorBoard(log_dir='.logs/disc', write_graph=False)
    tb_disc.set_model(d)
    # training

    if load_weights:
        g.load_weights(filepath='weights/generator.h5')
        d.load_weights(filepath='weights/discriminator.h5')
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            if index > 0:
                logger.setLevel(logging.INFO)
            noise = np.random.uniform(-1.0, 1.0, size=(BATCH_SIZE, 128))

            domains_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            logger.debug("domains_batch size")
            logger.debug(domains_batch.shape)

            generated_domains = g.predict(noise, verbose=0)
            logger.debug("generated domains shape:")
            logger.debug(generated_domains.shape)

            X = np.concatenate((domains_batch, generated_domains))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            d_loss = d.train_on_batch(X, y)
            d_log = ("batch %d\t[ DISC\tloss : %f ]" % (index, d_loss))
            noise = np.random.uniform(-1.0, 1.0, size=(BATCH_SIZE, 128))

            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            if index % 10 == 9:
                logger.info("%s\t[ GENR\tloss : %f ]" % (d_log, g_loss))
                write_log(callback=tb_gen, names=['loss'], logs=g_loss, batch_no=index // 10)
                write_log(callback=tb_disc, names=['loss'], logs=d_loss, batch_no=index // 10)
                g.save_weights('weights/generator.h5', True)
                d.save_weights('weights/discriminator.h5', True)

        generate(10, inv_map)


def write_log(callback, names, logs, batch_no):
    if isinstance(logs, list):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
    else:
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = logs
        summary_value.tag = names[0]
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def generate(BATCH_SIZE, inv_map):
    def to_readable_domain(decoded):
        domains = []
        for j in range(decoded.shape[0]):
            word = ""
            for i in range(decoded.shape[1]):
                if decoded[j][i] != 0:
                    word = word + inv_map[decoded[j][i]]
            domains.append(word)
        return domains

    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('weights/generator.h5')
    noise = np.random.uniform(-1, 1, (BATCH_SIZE, 128))
    generated_images = g.predict(noise, verbose=1)
    converted = K.eval(sampling(generated_images))
    readable = to_readable_domain(converted)
    import itertools
    for x, y in itertools.izip(converted, readable):
        logger.info("%s\t%s" % (x, y))


def build_dataset(n_samples=10000, maxlen=15):
    df = pd.DataFrame(
        pd.read_csv("../../dataset/legitdomains.txt",
                    sep=" ",
                    header=None,
                    names=['domain']),
        dtype=str)

    if n_samples:
        df = df.sample(n=n_samples, random_state=42)

    X_ = df['domain'].values

    # preprocessing text
    tk = Tokenizer(char_level=True)
    tk.fit_on_texts(string.lowercase + string.digits + '-' + '.')
    seq = tk.texts_to_sequences(X_)
    X = sequence.pad_sequences(seq, maxlen=maxlen)
    inv_map = {v: k for k, v in tk.word_index.iteritems()}
    X_tmp = []
    for x in X:
        X_tmp.append(to_categorical(x, tk.document_count))

    X = np.array(X_tmp)
    return X, tk.document_count, inv_map


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
        # generate(BATCH_SIZE=args.batch_size, nice=args.nice)
        pass
