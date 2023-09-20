from math import sqrt
from math import log
from numpy.random import randn
import numpy as np
from utils import *
from tqdm import tqdm

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
            #print("len", len(input_data[i]))
            if len(input_data[i]) != self.Layers[0].nb_neurons:
                print("model:predict wrong dimensions")
                return None
            
            pred = model.__activation(input_data[i] + self.Layers[0].bias, self.Layers[0].activation)
            #self.Layers[0].values = pred
            for idx in range(self.nb_layers - 1):
                pred = np.matmul(self.weight_matrices[idx], pred) + self.Layers[idx + 1].bias
                #print(self.Layers[idx + 1].activation)
                pred = model.__activation(pred, self.Layers[idx + 1].activation)
                #self.Layers[idx].values = pred
            res.append(pred)
        return np.array(res)

    def __forwarding(self, load):
        res = []
        if not isinstance(load, np.ndarray):
            print("model:forwarding type error")
        if len(load) != self.Layers[0].nb_neurons:
            print("model:forwarding wrong dimensions")
            return None

        pred = model.__activation(load + self.Layers[0].bias, self.Layers[0].activation)
        res.append(pred)
        for idx in range(self.nb_layers - 1):
            pred = np.matmul(self.weight_matrices[idx], pred) + self.Layers[idx + 1].bias
            pred = model.__activation(pred, self.Layers[idx + 1].activation)
            res.append(pred)
        return res

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



    def fit(self, network, data_train, data_valid, truth, loss='binaryCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84):
        for steps in range(epochs):


            for i in range(data_train.shape[0]):
                print(self.__forwarding(data_train[i]))
            Y_hat = self.predict(data_train)
            Y_vhat = self.predict(data_valid)

            loss = self.lossbce(Y_hat, truth)
            val_los = self.lossbce(Y_vhat, truth)

            print("epoch {}/{} - loss: {} - val_los : {}".format(steps, epochs, loss, val_los))
            # for upd in range(self.nb_layers - 1):
            #     (x, y) = (self.weight_matrices[upd].shape[0], self.weight_matrices[upd].shape[1])
            #     self.weight_matrices[upd] = sqrt(2.0 / x) * np.random.randn(x, y)

            #for k in range(last_layer.shape[0]):
            weightL = self.weight_matrices[self.nb_layers - 2]
            last_layer = self.Layers[self.nb_layers - 1]
            #print(last_layer.values[0], last_layer.values[1])
                


