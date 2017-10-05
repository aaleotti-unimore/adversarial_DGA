# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
import string

import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

# fix random seed for reproducibility
np.random.seed(7)
np.set_printoptions(linewidth=200)
# load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
n_samples = 10

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
df = pd.DataFrame(pd.read_csv("../../detect_DGA/datasets/legit_dga_domains.csv", sep=","))
if n_samples:
    df = df.sample(n=n_samples, random_state=42)
X_ = df['domain'].values
y = np.ravel(lb.fit_transform(df['class'].values))

tk = Tokenizer(char_level=True)

tk.fit_on_texts(string.lowercase + string.digits + '-')
print("word index: %s" % len(tk.word_index))
# print(tk.word_index)
seq = tk.texts_to_sequences(X_)
for x, s in zip(X_, seq):
    print(x, s)
print("")
print("X shape before padding: %s" % len(seq))
X = sequence.pad_sequences(seq)
print("X shape after padding: " + str(X.shape))
print(X)

model = Sequential()
# input_ = Input(shape=(None, X.shape[1]), dtype='int32', name='main_input')
# x = Embedding(output_dim=20, input_length=X.shape[1], input_dim=37)(input_)
# conv1 = Conv1D(20, 2, padding='valid', activation='relu', strides='1')(x)
# # conv2 = Conv1D(10, 3, padding='valid', activation='relu', strides='1')(x)
# # max1 = MaxPooling1D()(conv1)
# max2 = MaxPooling1D()(conv1)
# import keras
# # high = Model(inputs=[max1,max2])
# high = keras.layers.Highway()(conv1)
# lstm = LSTM()(high)
#
# model.add(input_)
# model.summary()
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout

inp = Input(shape=(X.shape[1],))
print(inp)
emb = Embedding(len(tk.word_index), 20, input_length=X.shape[1])(inp)
print(emb)
filters = [20, 10]
kernels = [2, 3]
convs = []
for i in range(2):
    conv = Conv1D(filters[i],
                  kernels[i],
                  padding='valid',
                  activation='relu',
                  strides=1)(emb)
    conv = Dropout(0.1)(conv)
    conv = MaxPooling1D()(conv)
    convs.append(conv)

from keras.layers.merge import concatenate

out = concatenate(convs)

conv_model = Model(inputs=inp, outputs=out)
model.add(conv_model)
from keras.layers import Highway
# model.add(Highway())
model.add(LSTM(128))
# model.fit(X,y)
#
# #
# # input_array = np.random.randint(1000, size=(32, 10))
# # model.add(Embedding(1000, 64, input_length=10))
model.compile('rmsprop', 'mse')
from keras.utils.vis_utils import plot_model

plot_model(model, show_shapes=True)
model.summary()

# output_array = model.predict(X)
# print(output_array.shape)
# print(model.predict(X[0]).shape)
