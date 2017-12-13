import argparse
import logging
import os
import string
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv1D, Dropout, concatenate, LSTM, RepeatVector, Dense, TimeDistributed, \
    LeakyReLU, BatchNormalization, Flatten, AveragePooling1D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import RMSprop, adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, plot_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CUDA_VISIBLE_DEVICES = 0


# K.set_learning_phase(0)
# print("set learning phase to %s" % K.learning_phase())


def generator_model(summary=True, print_fn=None):
    """
    Generator model:
    # In: (batch_size, 1),
    # Out: (batch_size, timesteps, word_index)
    :param summary: set to True to have a summary printed to output and a plot to file at "images/discriminator.png"
    :return: generator model
    """
    dropout_value = 0.4
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    cnn_strides = [1, 1]
    dec_convs = []
    leaky_relu_alpha = 0.2
    latent_vector = 38
    timesteps = 15
    word_index = 38

    dec_inputs = Input(shape=(latent_vector,),
                       name="Generator_Input")
    # decoded = Dense(word_index)(dec_inputs)
    # decoded = BatchNormalization(momentum=0.9)(decoded)
    # decoded = LeakyReLU(leaky_relu_alpha)(decoded)
    # decoded = Dropout(dropout_value)(decoded)
    decoded = RepeatVector(timesteps, name="gen_repeate_vec")(dec_inputs)
    decoded = LSTM(word_index, return_sequences=True, name="gen_LSTM")(decoded)
    decoded = Dropout(dropout_value)(decoded)
    for i in range(2):
        conv = Conv1D(cnn_filters[i],
                      cnn_kernels[i],
                      padding='same',
                      strides=cnn_strides[i],
                      name='gen_conv%s' % i)(decoded)
        # conv = BatchNormalization(momentum=0.9)(conv)
        conv = LeakyReLU(alpha=leaky_relu_alpha)(conv)
        conv = Dropout(dropout_value, name="gen_dropout%s" % i)(conv)
        dec_convs.append(conv)

    decoded = concatenate(dec_convs)
    decoded = TimeDistributed(Dense(word_index, activation='softmax'), name='decoder_end')(
        decoded)  # output_shape = (samples, maxlen, max_features )

    G = Model(inputs=dec_inputs, outputs=decoded, name='Generator')
    if summary:
        if print_fn:
            G.summary(print_fn=print_fn)
        G.summary()
    return G


def discriminator_model(summary=True, print_fn=None):
    """
    Discriminator model takes a 3D tensor of size (batch_size, timesteps, word_index), outputs a domain embedding tensor of size (batch_size, lstm_vec_dim).
    :param summary: set to True to have a summary printed to output and a plot to file at "images/discriminator.png"
    :return: Discriminator model
    """
    dropout_value = 0.4
    cnn_filters = [20, 10]
    cnn_kernels = [2, 3]
    cnn_strides = [1, 1]
    enc_convs = []
    embedding_vec = 20  # lunghezza embedding layer
    leaky_relu_alpha = 0.2
    timesteps = 15
    word_index = 38
    latent_vector = 38

    discr_inputs = Input(shape=(timesteps, word_index),
                         name="Discriminator_Input")
    # embedding layer. expected output ( batch_size, timesteps, embedding_vec)
    # manual_embedding = Dense(embedding_vec, activation='linear', name="manual_embedding")
    # discr = TimeDistributed(manual_embedding, name='embedded', trainable=False)(discr_inputs)
    # discr = Embedding(word_index, embedding_vec, input_length=timesteps, name="discr_embedd")(
    #     discr_inputs)  # other embedding layer
    for i in range(2):
        conv = Conv1D(cnn_filters[i],
                      cnn_kernels[i],
                      padding='same',
                      strides=cnn_strides[i],
                      name='discr_conv%s' % i)(discr_inputs)
        # conv = BatchNormalization(momentum=0.9)(conv)
        conv = LeakyReLU(alpha=leaky_relu_alpha)(conv)
        conv = Dropout(dropout_value, name='discr_dropout%s' % i)(conv)
        # conv = AveragePooling1D()(conv)
        enc_convs.append(conv)

    # concatenating CNNs. expected output (batch_size, 7, 30)
    discr = concatenate(enc_convs)
    discr = Flatten()(discr)
    # discr = LSTM(latent_vector)(discr)
    # discr = Dropout(dropout_value)(discr)
    discr = Dense(1, activation='sigmoid',
                  kernel_initializer='normal'
                  )(discr)

    D = Model(inputs=discr_inputs, outputs=discr, name='Discriminator')

    if summary:
        if print_fn:
            D.summary(print_fn=print_fn)
        else:
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
    adv_model = Sequential()
    adv_model.add(g)
    d.trainable = False
    adv_model.add(d)
    return adv_model


def train(BATCH_SIZE=32, disc=None, genr=None, original_model_name=None):
    """
    Training function.
    :param BATCH_SIZE
    """
    directory = os.path.join("experiments", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(directory):
        # crea la cartella
        os.makedirs(directory)
        os.makedirs(directory + "/model")

    hdlr = logging.FileHandler(os.path.join(directory, 'output.log'))
    logger.addHandler(hdlr)

    logger.debug(directory)
    if original_model_name is not None:
        logger.debug("MORE TRAINING on the model %s" % original_model_name)

    # load dataset
    latent_dim = 38
    maxlen = 15
    n_samples = int(10000 + 1000 * 0.33)
    data_dict = __build_dataset(maxlen=maxlen, n_samples=n_samples)
    X_train = data_dict['X_train']

    print("Training set shape %s" % (X_train.shape,))

    # models
    if disc is None:
        disc = discriminator_model(print_fn=logger.debug)
        plot_model(disc, to_file=os.path.join(directory, "discriminator.png"), show_shapes=True)
    if genr is None:
        genr = generator_model(print_fn=logger.debug)
        plot_model(genr, to_file=os.path.join(directory, "generator.png"), show_shapes=True)

    gan = adversarial(genr, disc)

    #   optimizers
    discr_opt = RMSprop(lr=0.0005,
                        clipvalue=1.0,
                        decay=1e-8)
    # gan_opt = RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8) #usual
    gan_opt = adam(lr=0.0004,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-8,
                   decay=1e-8,
                   clipvalue=1.0)  # alternative
    #   compilation
    gan.compile(loss='binary_crossentropy', optimizer=gan_opt)
    disc.trainable = True
    disc.compile(loss='binary_crossentropy', optimizer=discr_opt)
    gan.summary(print_fn=logger.debug)

    # callbacks
    tb_gan = TensorBoard(log_dir=os.path.join(directory, ".log/gan"), write_graph=True, histogram_freq=0,
                         batch_size=BATCH_SIZE)
    tb_gan.set_model(gan)
    tb_disc = TensorBoard(log_dir=os.path.join(directory, ".log/disc"), write_graph=True, histogram_freq=0,
                          batch_size=BATCH_SIZE)
    tb_disc.set_model(disc)

    batch_no = 0
    for epoch in range(300):
        logger.info("Epoch is %s" % epoch)
        logger.info("Number of batches %s" % int(X_train.shape[0] / BATCH_SIZE))
        logger.debug("Batch size: %s" % BATCH_SIZE)

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            if index > 0:
                # show debug data only the first time.
                logger.setLevel(logging.INFO)

            noise = np.random.normal(size=(BATCH_SIZE, latent_dim))  # random latent vectors. same size of
            alexa_domains = X_train[(index * BATCH_SIZE):(index + 1) * BATCH_SIZE]
            logger.debug("domains_batch size:\t%s" % (alexa_domains.shape,))

            # Generating domains from generator
            generated_domains = genr.predict(noise, verbose=0)
            logger.debug("generated domains shape:\t%s" % (generated_domains.shape,))

            # usual trainig mode
            # combined_domains = np.concatenate((domains_batch, generated_domains))
            # labels = np.concatenate([np.ones((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))]) # 1 = real, 0 = fake
            # labels += 0.05 * np.random.random(labels.shape)

            # alternative training mode:
            labels_size = (BATCH_SIZE, 1)
            labels_real = np.random.uniform(0.9, 1.1, size=labels_size)  # ~1 = real. Label Smoothing technique
            labels_fake = np.zeros(shape=labels_size)  # 0 = fake
            if index % 2 == 0:
                training_domains = alexa_domains
                labels = labels_real
            else:
                training_domains = generated_domains
                labels = labels_fake

            logger.debug("training set shape\t%s" % (training_domains.shape,))
            logger.debug("target shape %s" % (labels.shape,))

            # training discriminator on both alexa and generated domains
            disc.trainable = True
            disc_history = disc.train_on_batch(training_domains, labels)
            disc.trainable = False

            # training generator model inside the adversarial model
            noise = np.random.normal(size=(BATCH_SIZE, latent_dim))  # random latent vectors.
            misleading_targets = np.random.uniform(0.9, 1.1, size=labels_size)
            gan_history = gan.train_on_batch(noise, misleading_targets)

            if index % 10 == 9:
                d_log = ("epoch %d\tmini-batch %d\t[ DISC\tloss : %f ]" % (epoch, index, disc_history))
                logger.info("%s\t[ ADV\tloss : %f ]" % (d_log, gan_history))
                __write_log(callback=tb_gan,
                            names=gan.metrics_names,
                            logs=gan_history,
                            batch_no=batch_no)
                __write_log(callback=tb_disc,
                            names=disc.metrics_names,
                            logs=disc_history,
                            batch_no=batch_no)
                gan.save(os.path.join(directory, 'model/gan.h5'))
                disc.save(os.path.join(directory, 'model/discriminator.h5'))
                genr.save(os.path.join(directory, 'model/generator.h5'))
                generate(generated_domains, inv_map=data_dict['inv_map'])

            batch_no += 1


def generate(predictions, inv_map=None, n_samples=5, temperature=1.0, print_preds=False):
    if inv_map is None:
        datas_dict = __build_dataset()
        inv_map = datas_dict['inv_map']

    sampled = []
    for x in predictions[:n_samples]:
        word = []
        for y in x:
            word.append(__np_sample(y, temperature=temperature))
        sampled.append(word)

    readable = __to_readable_domain(np.array(sampled), inv_map=inv_map)
    if print_preds:
        import itertools
        for s, r in itertools.izip(sampled, readable):
            logger.info("%s\t%s" % (s, r))
    else:
        logger.info("Generated sample: %s " % readable)


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
              # batch_size=32,
              epochs=500)

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


def __custom_gan_loss(y_true, y_pred):
    return -(K.max(K.log(y_pred)))


def __build_dataset(n_samples=10000, maxlen=15, validation_split=0.33):
    df = pd.DataFrame(
        pd.read_csv("resources/datasets/legitdomains.txt",
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
    # preds = np.random.multinomial(1, preds, 1)
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    if args.mode == "moretrain":
        disc = load_model("experiments/20171214-115137/model/discriminator.h5")
        genr = load_model("experiments/20171214-115137/model/generator.h5")
        train(BATCH_SIZE=args.batch_size, disc=disc, genr=genr, original_model_name="20171214-115137")
    elif args.mode == "generate":
        model = load_model("experiments/20171213-112910/model/discriminator.h5")
        model.summary()
        emb_weights = list(model.layers)
        for e in emb_weights:
            print(e)
        # preds = model.predict_on_batch(np.random.normal(size=(args.batch_size, 38)))
        # generate(predictions=preds, n_samples=args.batch_size, temperature=1.0, print_preds=True)
        pass
    elif args.mode == "autoencoder":
        train_autoencoder()
    elif args.mode == "test-autoencoder":
        test_autoencoder()
