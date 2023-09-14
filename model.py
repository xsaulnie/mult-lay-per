from math import sqrt
from numpy.random import randn
import numpy as np

class model():
    def __init__(self, nb_layers , weight_matrices):
        self.nb_layers = nb_layers
        self.weight_matrices = weight_matrices
    def createNetwork(listLayers):
        weight_matrices = []

        for idx in range (len(listLayers) - 1):
           weight_matrices.append(model.__initialize_weight(listLayers[idx].nb_neurons, listLayers[idx+1].nb_neurons, listLayers[idx].weights_initializer))
        return (model(len(listLayers), weight_matrices))

    def __initialize_weight(nb_input, nb_output, weights_initializer):
        if weights_initializer == "heUniform":
            return sqrt(2.0 / nb_input) * np.random.randn(nb_input, nb_output)

        return res