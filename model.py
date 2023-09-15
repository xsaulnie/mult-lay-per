from math import sqrt
from math import log
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
            print("model.predict : bad argument")
            return None
        res = []
        for i in range(input_data.shape[0]):
            print("len", len(input_data[i]))
            if len(input_data[i]) != self.Layers[0].nb_neurons:
                print("model:predict wrong dimensions")
                return None
            
            pred = model.__activation(input_data[i], self.Layers[0].activation)
            for idx in range(self.nb_layers - 1):
                pred = np.matmul(self.weight_matrices[idx], pred) + self.Layers[idx].bias
                print(self.Layers[idx + 1].activation)
                pred = model.__activation(pred, self.Layers[idx + 1].activation)
            res.append(pred)
        return np.array(res)

    def lossmse(self, y_hat, y):
        if not isinstance(y_hat, np.ndarray) or not isinstance(y, np.ndarray):
            print("model.lossmse : bad argument")
            return None
        if y_hat.shape[0] != y.shape[0]:
            print("model.lossmse : bad dimensions")
            return None
        return ((sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0]))[0])
    
    def lossbce(self, y_hat, y):
        if not isinstance(y_hat, np.ndarray) or not isinstance(y, np.ndarray):
            print("model.lossbce : bad argument")
            return None
        if y_hat.shape[0] != y.shape[0]:
            print("model.lossbce : bad argument")
            return None
        ret = 0
        for idx in range(y.shape[0]):
            ret = ret + (y[idx][0] * math.log(y_hat[idx][0] + 1e-15) + (1 - y[idx][0])* math.log(1 - y_hat[idx][0] + 1e-15))
            ret = ret + (y[idx][1] * math.log(y_hat[idx][1] + 1e-15) + (1 - y[idx][1])* math.log(1 - y_hat[idx][1] + 1e-15))
        return (- ret / 2 * y.shape[0])

    def fit(self, network, data_train, data_valid, loss='binaryCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84):
        

