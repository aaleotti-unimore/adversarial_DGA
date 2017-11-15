import numpy as np
from keras import backend as K
from keras.layers import Lambda, Input
from keras.models import Model


def original_sampling_fnc(preds):
    def __sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float32')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    domains = []
    for j in range(preds.shape[0]):
        word = []
        for i in range(preds.shape[1]):
            word.append(__sample(preds[j][i]))
        domains.append(word)

    return domains


def sampling2(tensor):
    def __sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        # preds = K.expand_dims(preds,axis=0)
        # print(preds)
        preds = K.log(preds) / temperature
        exp_preds = K.exp(preds)
        preds = exp_preds / K.sum(exp_preds)
        probas = np.random.multinomial(1, K.eval(preds), 1)
        return K.expand_dims(K.variable(np.argmax(probas)))

    batch_size, y_dim, z_dim = K.int_shape(tensor)
    for i in range(1):
        result = __sample(tensor[i, 0, :])
        for j in range(y_dim):
            if j == 0:
                continue
            c = __sample(tensor[i, j, :])
            result = K.concatenate([result, c], axis=0)

    return result


if __name__ == '__main__':
    K.set_learning_phase(0)

    tens = K.random_uniform(shape=(2, 15, 38), maxval=1, minval=0, dtype="float32", seed=42)
    inp = Input(shape=(15, 38),
                tensor=tens
                )
    exa = Lambda(lambda x: sampling2(x), output_shape=(15,))(inp)
    model = Model(inputs=inp, outputs=exa)
    model.summary()

    noise = np.random.uniform(0, 1, size=(2, 15, 38))
    noise = None
    print(model.predict_on_batch(noise))
