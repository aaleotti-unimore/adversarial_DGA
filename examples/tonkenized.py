import string

import numpy as np
import pandas as pd
from keras import Input
from keras import backend as K
from keras.layers import Conv1D, Dropout, MaxPooling1D, concatenate, LSTM, RepeatVector, Dense, Lambda, Embedding, \
    TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, one_hot
from keras.utils import plot_model, to_categorical
from sampling_layer import Sampling

n_samples = 2
maxlen = 15

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
# print("word index: %s" % len(tk.word_index))
seq = tk.texts_to_sequences(X_)
#
print(X_)
# caz = tk.texts_to_matrix(X_,mode='binary')
# print(seq)
# print(caz)




# for x, s in zip(X_, seq):
#     print(x, s)
# print("")
X = sequence.pad_sequences(seq, maxlen=maxlen)
print("X shape after padding: " + str(X.shape))
# print(X)
X1 = []
for x in X:
    X1.append(to_categorical(x, 38))

X = np.array(X1)
print(X.shape)


