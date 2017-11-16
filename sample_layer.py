import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model


def sampling(preds, temperature=1.0):
    preds = K.log(preds) / temperature
    exp_preds = K.exp(preds)
    preds = exp_preds / K.sum(exp_preds)
    return K.argmax(preds, axis=2)


if __name__ == '__main__':
    K.set_learning_phase(1)
    batch_size = 10

    ########### MODEL
    inp = Input(shape=(15, 38))
    exa = Lambda(lambda x: sampling(x), output_shape=(15,))(inp)
    model = Model(inputs=inp, outputs=exa)
    model.summary()

    noise = np.random.uniform(0, 1, size=(batch_size, 15, 38))
    print("NOISE INPUT")
    print(noise)
    decoded = model.predict_on_batch(noise)
    print("\n\nSAMPLED OUTPUT")
    print(decoded)
    #################



    #############  decoding into readable domain
    # tk = Tokenizer(char_level=True)
    # tk.fit_on_texts(string.lowercase + string.digits + '-' + '.')
    # inv_map = {v: k for k, v in tk.word_index.iteritems()}
    #
    # domains = []
    #
    # for j in range(decoded.shape[0]):
    #     word = ""
    #     for i in range(decoded.shape[1]):
    #         if decoded[j][i] != 0:
    #             word = word + inv_map[decoded[j][i]]
    #     domains.append(word)
    #
    # domains = np.char.array(domains)
    # for domain in domains:
    #     print(domain)
