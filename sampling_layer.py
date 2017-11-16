import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model


class Sampling(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Sampling, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Sampling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        temperature = 1.0
        x = K.log(x) / temperature
        exp_preds = K.exp(x)
        x = exp_preds / K.sum(exp_preds)

        return K.argmax(x, axis=2)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)






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
