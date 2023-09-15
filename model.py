from math import sqrt
from numpy.random import randn
import numpy as np
from utils import *

class model():
    def __init__(self, nb_layers , weight_matrices, Layers):
        self.nb_layers = nb_layers
        self.weight_matrices = weight_matrices
        self.Layers = Layers

    def createNetwork(listLayers):
        weight_matrices = []

        for idx in range (len(listLayers) - 1):
           weight_matrices.append(model.__initialize_weight(listLayers[idx+1].nb_neurons, listLayers[idx].nb_neurons, listLayers[idx].weights_initializer))
        return (model(len(listLayers), weight_matrices, listLayers))

    def __initialize_weight(nb_input, nb_output, weights_initializer):
        if weights_initializer == "heUniform":
            return sqrt(2.0 / nb_input) * np.random.randn(nb_input, nb_output)
        return res

    def __activation(array, activation_type):
        if activation_type == 'sigmoid':
            vfunc = np.vectorize(sigmoid)
        if activation_type == 'softmax':
            return softmax(array)
        return vfunc(array)

    def predict(self, input_data):
        if (not isinstance(input_data, np.ndarray)):
            print("model:predict bad argument")
            return None
        if input_data.shape[1] != 1 or input_data.shape[0] != self.Layers[0].nb_neurons:
            print("model:predict wrong dimensions")
            return None
        pred = model.__activation(input_data, self.Layers[0].activation)
        for idx in range(self.nb_layers - 1):
            pred = np.matmul(self.weight_matrices[idx], pred) + self.Layers[idx].bias
            print(self.Layers[idx + 1].activation)
            pred = model.__activation(pred, self.Layers[idx + 1].activation)
        return pred