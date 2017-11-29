import argparse
import logging
import string

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Conv1D, Dropout, MaxPooling1D, concatenate, LSTM, RepeatVector, Dense, TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model, to_categorical

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CUDA_VISIBLE_DEVICES = 0


# K.set_learning_phase(0)
# print("set learning phase to %s" % K.learning_phase())


def generator_model(summary=None):
    """
    Generator model:
    # In: (batch_size, 128),
    # Out: (batch_size, timesteps, word_index)
    :param summary: set to True to have a summary printed to output and a plot to file at "images/discriminator.png"
    :return: generator model
    """
    dropout_value = 0.1
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    dec_convs = []
    lstm_vec_dim = 128
    timesteps = 15
    word_index = 38

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
    """
    Discriminator model takes a 3D tensor of size (batch_size, timesteps, word_index), outputs a domain embedding tensor of size (batch_size, lstm_vec_dim).
    :param summary: set to True to have a summary printed to output and a plot to file at "images/discriminator.png"
    :return: Discriminator model
    """
    dropout_value = 0.1
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    enc_convs = []
    embedding_vec = 20  # lunghezza embedding layer
    timesteps = 15
    word_index = 38
    lstm_vec_dim = 128
    # In: (batch_size, timesteps, word_index),
    # Out: (batch_size, 128)

    discr_inputs = Input(shape=(timesteps, word_index),
                         name="Discriminator_Input")
    # embedding layer. expected output ( batch_size, timesteps, embedding_vec)
    manual_embedding = Dense(embedding_vec, activation='linear')
    discr = TimeDistributed(manual_embedding, name='manual_embedding', trainable=False)(
        discr_inputs)
    # discr = Embedding(word_index, embedding_vec, input_length=timesteps)(discr_inputs) #other embedding layer
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


def generator_containing_discriminator(g, d):
    """
    Adversarial Model
    :param g: Generator
    :param d: Discriminator
    :return: Adversarial model
    """
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def train(BATCH_SIZE, load_weights=True, ):
    """
    Training function.
    :param BATCH_SIZE:
    :param load_weights: preload weights from weights folder
    """

    # load dataset
    maxlen = 15
    n_samples = 100000
    X_train, word_index, inv_map = __build_dataset(maxlen=maxlen, n_samples=n_samples)
    print("Training set shape %s" % (X_train.shape,))
    print("Word index: %s" % word_index)

    # models
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    #   optimizers
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    #   compilation
    g.compile(loss='binary_crossentropy', optimizer="SGD")  # compiling generator
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim, metrics=['accuracy'])
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])

    # callbacks
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

            domains_batch = X_train[(index * BATCH_SIZE):(index + 1) * BATCH_SIZE]
            logger.debug("domains_batch size:\t%s" % (domains_batch.shape,))

            noise = np.random.uniform(-1.0, 1.0, size=(BATCH_SIZE, 128))
            generated_domains = g.predict(noise, verbose=0)
            logger.debug("generated domains shape:\t%s" % (generated_domains.shape,))

            X = np.concatenate((domains_batch, generated_domains))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            logger.debug("training set shape\t%s" % (X.shape,))
            logger.debug("target shape %s" % (y.shape,))

            d_loss = d.train_on_batch(X, y)
            d_log = ("batch %d\t[ DISC\tloss : %f\tacc : %f ]" % (index, d_loss[0], d_loss[1]))
            noise = np.random.uniform(-1.0, 1.0, size=(BATCH_SIZE, 128))

            d.trainable = False
            d_on_g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            if index % 10 == 9:
                logger.info("%s\t[ GENR\tloss : %f\tacc: %f ]" % (d_log, d_on_g_loss[0], d_on_g_loss[1]))
                __write_log(callback=tb_gen, names=['loss'], logs=d_on_g_loss, batch_no=index // 10)
                __write_log(callback=tb_disc, names=['loss'], logs=d_loss, batch_no=index // 10)
                g.save_weights('weights/generator.h5', True)
                d.save_weights('weights/discriminator.h5', True)

        generate(10, inv_map)


def generate(BATCH_SIZE, inv_map=None):
    def to_readable_domain(decoded):
        domains = []
        for j in range(decoded.shape[0]):
            word = ""
            for i in range(decoded.shape[1]):
                if decoded[j][i] != 0:
                    word = word + inv_map[decoded[j][i]]
            domains.append(word)
        return domains

    if inv_map is None:
        _, __, inv_map = __build_dataset()

    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('weights/generator.h5')
    noise = np.random.uniform(-1, 1, (BATCH_SIZE, 128))
    generated_images = g.predict(noise, verbose=1)
    converted = K.eval(__sampling(generated_images))
    readable = to_readable_domain(converted)
    import itertools

    for x, y in itertools.izip(converted, readable):
        logger.info("%s\t%s" % (x, y))


def __build_dataset(n_samples=10000, maxlen=15):
    df = pd.DataFrame(
        pd.read_csv("../../dataset/legitdomains.txt",
                    sep=" ",
                    header=None,
                    names=['domain']),
        dtype=str)

    if n_samples:
        df = df.sample(n=n_samples, random_state=42)

    X_ = df.loc[df['domain'].str.len() > 5].values
    # X_ = X_.values
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


def __sampling(preds, temperature=1.0):
    """
     helper function to sample an index from a probability array

    :param preds: predictions data. 3D tensor of floats
    :param temperature: temperature
    :return: sampled data. 2D tensor of integers
    """
    preds = K.log(preds) / temperature
    exp_preds = K.exp(preds)
    preds = exp_preds / K.sum(exp_preds)
    return K.argmax(preds)


def __write_log(callback, names, logs, batch_no):
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
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
        generate(BATCH_SIZE=args.batch_size)
        pass
