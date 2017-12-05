import argparse
import logging
import string
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv1D, Dropout, MaxPooling1D, concatenate, LSTM, RepeatVector, Dense, TimeDistributed, \
    LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CUDA_VISIBLE_DEVICES = 0


# K.set_learning_phase(0)
# print("set learning phase to %s" % K.learning_phase())



def generator_model(summary=None):
    """
    Generator model:
    # In: (batch_size, 1),
    # Out: (batch_size, timesteps, word_index)
    :param summary: set to True to have a summary printed to output and a plot to file at "images/discriminator.png"
    :return: generator model
    """
    dropout_value = 0.3
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    cnn_strides = 1
    dec_convs = []
    lstm_vec_dim = 64
    timesteps = 15
    word_index = 38

    dec_inputs = Input(shape=(lstm_vec_dim,),
                       name="Generator_Input")
    # decoded = Dense(lstm_vec_dim)(dec_inputs)
    # decoded = LeakyReLU()(decoded)
    decoded = RepeatVector(timesteps, name="gen_repeate_vec")(
        dec_inputs)  # Repeating input by "timesteps" times. expected output (batch_size, 128, 15)
    decoded = LSTM(lstm_vec_dim, return_sequences=True, name="gen_LSTM")(decoded)

    for i in range(2):
        conv = Conv1D(cnn_filters[i],
                      cnn_kernels[i],
                      padding='same',
                      strides=cnn_strides,
                      name='gen_conv%s' % i)(decoded)
        conv = LeakyReLU()(conv)
        conv = Dropout(dropout_value, name="gen_dropout%s" % i)(conv)
        dec_convs.append(conv)

    decoded = concatenate(dec_convs)
    decoded = TimeDistributed(Dense(word_index, activation='softmax'), name='decoder_end')(
        decoded)  # output_shape = (samples, maxlen, max_features )

    G = Model(inputs=dec_inputs, outputs=decoded, name='Generator')
    if summary:
        G.summary()
        # plot_model(G, to_file="images/generator.png", show_shapes=True)
    return G


def discriminator_model(summary=None):
    """
    Discriminator model takes a 3D tensor of size (batch_size, timesteps, word_index), outputs a domain embedding tensor of size (batch_size, lstm_vec_dim).
    :param summary: set to True to have a summary printed to output and a plot to file at "images/discriminator.png"
    :return: Discriminator model
    """
    dropout_value = 0.3
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    cnn_strides = 1
    enc_convs = []
    embedding_vec = 20  # lunghezza embedding layer
    timesteps = 15
    word_index = 38
    lstm_vec_dim = 64

    discr_inputs = Input(shape=(timesteps, word_index),
                         name="Discriminator_Input")
    # embedding layer. expected output ( batch_size, timesteps, embedding_vec)
    manual_embedding = Dense(embedding_vec, activation='linear')
    discr = TimeDistributed(manual_embedding, name='manual_embedding', trainable=False)(discr_inputs)
    # discr = Embedding(word_index, embedding_vec, input_length=timesteps, name="discr_embedd")(
    #     discr_inputs)  # other embedding layer
    for i in range(2):
        conv = Conv1D(cnn_filters[i],
                      cnn_kernels[i],
                      padding='same',
                      strides=cnn_strides,
                      name='discr_conv%s' % i)(discr)
        conv = LeakyReLU()(conv)
        conv = Dropout(dropout_value, name='discr_dropout%s' % i)(conv)
        conv = MaxPooling1D()(conv)
        enc_convs.append(conv)

    # concatenating CNNs. expected output (batch_size, 7, 30)
    discr = concatenate(enc_convs)
    discr = LSTM(lstm_vec_dim)(discr)
    # discr = Dense(1, activation='sigmoid',
    #               kernel_initializer='normal'
    #               )(discr)

    D = Model(inputs=discr_inputs, outputs=discr, name='Discriminator')

    if summary:
        D.summary()
        # plot_model(D, to_file="images/discriminator.png", show_shapes=True)
    return D


def adversarial(g, d):
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


def train_autoencoder():
    data_dict = __build_dataset(n_samples=10000)

    directory = os.path.join("autoencoder_experiments", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(directory):
        # crea la cartella
        os.makedirs(directory)
        os.makedirs(directory + "/weights")

    d = discriminator_model()
    g = generator_model()
    model = Sequential()
    model.add(d)
    model.add(g)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(data_dict['X_train'], data_dict['X_train'],
              verbose=2,
              callbacks=[TensorBoard(log_dir=os.path.join(directory, ".logs"),
                                     histogram_freq=0,
                                     write_graph=0),
                         ModelCheckpoint(os.path.join(directory, "weights/autoencoder.h5"),
                                         monitor='val_loss',
                                         verbose=2,
                                         save_best_only=True, mode='auto')
                         ],
              validation_split=0.33,
              epochs=300)

    split = 50
    print("X_test")
    print(data_dict['X_test'][:split])

    predictions = model.predict(data_dict['X_test'][:split], verbose=0)
    sampled = []
    for x in predictions:
        word = []
        for y in x:
            word.append(__np_sample(y))
        sampled.append(word)

    print("results")
    readable = __to_readable_domain(np.array(sampled), inv_map=data_dict['inv_map'])
    for r in readable:
        print(r)


def test_autoencoder():
    data_dict = __build_dataset(n_samples=1000)
    d = discriminator_model()
    g = generator_model()
    model = Sequential()
    model.add(d)
    model.add(g)
    model.load_weights("autoencoder_experiments/20171205-185323/weights/autoencoder.h5")
    split = 50
    print("X_test")
    print(data_dict['X_test'][:split])

    predictions = model.predict(data_dict['X_test'][:split], verbose=0)
    sampled = []
    for x in predictions:
        word = []
        for y in x:
            word.append(__np_sample(y))
        sampled.append(word)

    print("results")
    readable = __to_readable_domain(np.array(sampled), inv_map=data_dict['inv_map'])
    for r in readable:
        print(r)


def train(BATCH_SIZE):
    """
    Training function.
    :param BATCH_SIZE
    """
    directory = os.path.join("experiments", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(directory):
        # crea la cartella
        os.makedirs(directory)
        os.makedirs(directory + "/weights")

    # load dataset
    maxlen = 15
    n_samples = 10000
    data_dict = __build_dataset(maxlen=maxlen, n_samples=n_samples)
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']

    print("Training set shape %s" % (X_train.shape,))

    # models
    d = discriminator_model()
    g = generator_model()
    adv = adversarial(g, d)

    #   optimizers
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    adv_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    #   compilation
    g.compile(loss='categorical_crossentropy', optimizer=d_optim)  # compiling generator
    adv.compile(loss='binary_crossentropy', optimizer=adv_optim, metrics=['accuracy'])
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])
    adv.summary()

    # callbacks
    tb_adv = TensorBoard(log_dir=os.path.join(directory, ".log/adv"), write_graph=False)
    tb_adv.set_model(g)
    tb_disc = TensorBoard(log_dir=os.path.join(directory, ".log/disc"), write_graph=False)
    tb_disc.set_model(d)

    batch_no = 0
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            if index > 0:
                logger.setLevel(logging.INFO)
            noise = np.random.normal(0.0, 1.0, size=(BATCH_SIZE, 1))

            domains_batch = X_train[(index * BATCH_SIZE):(index + 1) * BATCH_SIZE]
            logger.debug("domains_batch size:\t%s" % (domains_batch.shape,))

            # Generating domains from generator
            generated_domains = g.predict(noise, verbose=0)
            logger.debug("generated domains shape:\t%s" % (generated_domains.shape,))

            X = np.concatenate((domains_batch, generated_domains))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            logger.debug("training set shape\t%s" % (X.shape,))
            logger.debug("target shape %s" % (y.shape,))

            # training discriminator on both alexa and generated domains
            d_history = d.train_on_batch(X, y)
            d_log = ("batch %d\t[ DISC\tloss : %f\tacc : %f ]" % (index, d_history[0], d_history[1]))

            # training generator model inside the adversarial model
            d.trainable = False
            adv_history = adv.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            adv_test_history = adv.test_on_batch(noise,
                                                 [1] * noise.shape[0])
            if index % 10 == 9:
                logger.info("%s\t[ ADV\tloss : %f\tacc: %f ]" % (d_log, adv_history[0], adv_history[1]))
                __write_log(callback=tb_adv,
                            names=adv.metrics_names,
                            logs=adv_history,
                            batch_no=batch_no)
                __write_log(callback=tb_adv,
                            names=['test_loss', 'test_acc'],
                            logs=adv_test_history,
                            batch_no=batch_no)
                __write_log(callback=tb_disc,
                            names=d.metrics_names,
                            logs=d_history,
                            batch_no=batch_no)
                batch_no += 1

                g.save_weights(os.path.join(directory, 'weights/generator.h5'), True)
                d.save_weights(os.path.join(directory, 'weights/discriminator.h5'), True)

        generate(g=g, BATCH_SIZE=20, inv_map=data_dict['inv_map'])


def generate(g, BATCH_SIZE, inv_map=None):
    if inv_map is None:
        datas_dict = __build_dataset()
        inv_map = datas_dict['inv_map']

    noise = np.random.normal(0.0, 1.0, (BATCH_SIZE, 1))
    generated_domains = g.predict(noise, verbose=0)
    converted = K.eval(__sampling(generated_domains, temperature=0.5))
    readable = __to_readable_domain(converted)
    import itertools

    for x, y in itertools.izip(converted, readable):
        logger.info("%s\t%s" % (x, y))


def __build_dataset(n_samples=10000, maxlen=15, validation_split=0.33):
    df = pd.DataFrame(
        pd.read_csv("../../dataset/legitdomains.txt",
                    sep=" ",
                    header=None,
                    names=['domain'],
                    ))
    df = df.loc[df['domain'].str.len() > 5]
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
    return {'X_train': X[int(X.shape[0] * validation_split):, :, :],
            "X_test": X[:int(X.shape[0] * validation_split), :, :],
            "word_index": tk.document_count,
            "inv_map": inv_map}


def __np_sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # probas = np.random.multinomial(1, preds, 1)
    return np.argmax(preds)


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
    return K.argmax(preds, axis=2)


def __to_readable_domain(decoded, inv_map):
    domains = []
    for j in range(decoded.shape[0]):
        word = ""
        for i in range(decoded.shape[1]):
            if decoded[j][i] != 0:
                word = word + inv_map[decoded[j][i]]
        domains.append(word)
    return domains


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
    parser.add_argument("--batch-size", type=int, default=64)
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
    elif args.mode == "autoencoder":
        train_autoencoder()
    elif args.mode == "test-autoencoder":
        test_autoencoder()
