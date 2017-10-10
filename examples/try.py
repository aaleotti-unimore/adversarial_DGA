# LSTM and CNN for sequence classification in the IMDB dataset
import string

import numpy as np
import pandas as pd
from pandas import read_csv
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import LSTM, RepeatVector
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelBinarizer

# fix random seed for reproducibility
np.random.seed(7)
np.set_printoptions(linewidth=2000, threshold=100000)
# load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
n_samples = 1000

# loading db
lb = LabelBinarizer()
df = pd.DataFrame(pd.read_csv("../dataset/legitdomains.txt", sep=" ", header=None, names=['domain']))
if n_samples:
    df = df.sample(n=n_samples, random_state=42)
X_ = df['domain'].values
# y = np.ravel(lb.fit_transform(df['class'].values))

# preprocessing text
tk = Tokenizer(char_level=True)
tk.fit_on_texts(string.lowercase + string.digits + '-'+'.')
print("word index: %s" % len(tk.word_index))
seq = tk.texts_to_sequences(X_)
for x, s in zip(X_, seq):
    print(x, s)
print("")
X = sequence.pad_sequences(seq)
print("X shape after padding: " + str(X.shape))
print(X)

# autoencoder params
timesteps = X.shape[1]
filters = [20, 10]
kernels = [2, 3]
enc_convs = []
dec_convs = []
d = 20
###

enc_inputs = Input(shape=(X.shape[1],))
encoded = Embedding(len(tk.word_index), d, input_length=X.shape[1])(enc_inputs)
for i in range(2):
    conv = Conv1D(filters[i],
                  kernels[i],
                  padding='same',
                  activation='relu',
                  strides=1)(encoded)
    conv = Dropout(0.1)(conv)
    conv = MaxPooling1D()(conv)
    enc_convs.append(conv)

encoded = concatenate(enc_convs)
encoded = LSTM(128)(encoded)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(128, return_sequences=True)(decoded)

for i in range(2):
    conv = Conv1D(filters[i],
                  kernels[i],
                  padding='same',
                  activation='relu',
                  strides=1)(decoded)
    conv = Dropout(0.1)(conv)
    # conv = MaxPooling1D()(conv)
    dec_convs.append(conv)

decoded = concatenate(dec_convs)
decoded = Dense(len(tk.word_index), activation='sigmoid')(decoded)

autoencoder = Model(inputs=enc_inputs, outputs=decoded)

# optimizer = RMSprop(lr=0.01)
# autoencoder.compile(loss='categorical_crossentropy', optimizer=optimizer)
autoencoder.summary()
plot_model(autoencoder, to_file="autoencoder.png", show_shapes=True)
preds = autoencoder.predict(x=X, verbose=2)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float32')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


inv_map = {v: k for k, v in tk.word_index.iteritems()}
domains = []
word = ""
# print(inv_map)
for j in range(preds.shape[0]):
    word = ""
    for i in range(preds.shape[1]):
        k = sample(preds[j][i])
        if k > 0:
            word = word + inv_map[k]
    domains.append(word)

print(domains)

from detect_DGA import MyClassifier

rndf = MyClassifier(directory="/home/archeffect/PycharmProjects/detect_DGA/models/RandomForest tra:sup tst:sup")
true = np.ravel(np.zeros(len(domains), dtype=int))
res = rndf.predict(domains, true)
