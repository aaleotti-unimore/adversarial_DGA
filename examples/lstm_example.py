'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
import tensorflow as tf
import os
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

generator_path = '../dataset/xaa'
lstm_path = "../saved_models/lstm2"

dirtemp = os.path.join(lstm_path, "tensorboard")

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(generator_path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 15
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


callbacks = [
    TensorBoard(log_dir=dirtemp,
                write_graph=False,
                write_images=False,
                histogram_freq=0),
]

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    model.fit(X, y,
              validation_split=0.10,
              # metrics=['accuracy'],
              # callbacks=callbacks,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    json_model = model.to_json()
    dirmod = os.path.join(lstm_path, 'model_architecture.json')
    open(dirmod, 'w').write(json_model)
    model.save_weights(os.path.join(lstm_path, 'model_weitghts.h5'), overwrite=True)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: ' + sentence)
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            print(x)
            preds = model.predict(x, verbose=0)[0]
            print(preds)
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            print(generated)
            sentence = sentence[1:] + next_char
            print(sentence)

            sys.stdout.write(next_char)
            sys.stdout.flush()

        print()

        with open(os.path.join(lstm_path, "generated.txt"), 'a') as f:
            f.write(generated)
