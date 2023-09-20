import numpy as np
class layers():
    def __init__(self, nb_neurons, activation, weights_initializer, bias, type_layer, values='null'):
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.bias = bias
        self.type_layer = type_layer
        if values == 'null':
            self.values = np.array([0] * self.nb_neurons)

    def DenseLayer(nb_neurons, activation='sigmoid', weights_initializer='heUniform', bias = 0):
        return layers(nb_neurons,activation, weights_initializer, bias ,'DenseLayer')