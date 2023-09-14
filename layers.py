class layers():
    def __init__(self, nb_neurons, activation, weights_initializer, bias, type_layer):
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.bias = bias
        self.type_layer = type_layer

    def DenseLayer(nb_neurons, activation='sigmoid', weights_initializer='heUniform', bias = 0):
        return layers(nb_neurons,activation, weights_initializer, bias ,'DenseLayer')