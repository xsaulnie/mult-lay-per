from math import sqrt
from math import log
from numpy.random import randn
import numpy as np
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

class model():
    def __init__(self, nb_layers , weight_matrices, bias, Layers):
        self.nb_layers = nb_layers
        self.weight_matrices = weight_matrices
        self.Layers = Layers
        self.bias = bias

    def createNetwork(listLayers):
        weight_matrices = []
        bias = []

        for idx in range (len(listLayers) - 1):
            weight_matrices.append(model.__initialize_weight(listLayers[idx+1].nb_neurons, listLayers[idx].nb_neurons, listLayers[idx+1].weights_initializer))
            bias.append(np.zeros(listLayers[idx+1].nb_neurons))

        return (model(len(listLayers), weight_matrices, bias, listLayers))

    def __create_minigrad(self):
        grad_weight = []
        grad_bias = []
        for idx in range(self.nb_layers - 1):
            grad_weight.append(np.zeros((self.Layers[idx + 1].nb_neurons, self.Layers[idx].nb_neurons)))
            grad_bias.append(np.zeros(self.Layers[idx+1].nb_neurons))
        return (grad_weight, grad_bias)

    def __initialize_weight(nb_input, nb_output, weights_initializer):
        np.random.seed(42)
        print(weights_initializer)
        if weights_initializer == "heUniform":
            return sqrt(2.0 / nb_input) * np.random.randn(nb_input, nb_output)
        if weights_initializer == "zero":
            return np.ones((nb_input, nb_output))
        return res

    def __activation(array, activation_type):
        if activation_type == 'sigmoid':
            vfunc = np.vectorize(sigmoid)
        if activation_type == 'softmax':
            return softmax(array)
        if activation_type == 'relu':
            return relu(array)
        return vfunc(array)

    def __derivative(array, function_type):
        #print("derivate()", array, function_type)
        if function_type == 'relu':
            return np.array([1 if i > 0 else 0 for i in array])
        if function_type == 'sigmoid':
            return np.array([ i * (1 - i) for i in array])

    def predict(self, input_data):
        if (not isinstance(input_data, np.ndarray)):
            print("model.predict : bad argument")
            return None
        if input_data.shape[1] != self.Layers[0].nb_neurons:
            print("model:predict wrong dimensions")
            return None
        res = []
        for i in range(input_data.shape[0]):
            
            pred = model.__activation(input_data[i], self.Layers[0].activation)
            for idx in range(self.nb_layers - 1):
                pred = np.matmul(self.weight_matrices[idx], pred) + self.bias[idx]
                pred = model.__activation(pred, self.Layers[idx + 1].activation)
            res.append(pred)
        return np.array(res)

    def __forwarding(self, load):
        res = []
        if not isinstance(load, np.ndarray):
            print("model:forwarding type error")
        if len(load) != self.Layers[0].nb_neurons:
            print("model:forwarding wrong dimensions")
            return None

        pred = model.__activation(load, self.Layers[0].activation)
        res.append(pred)
        for idx in range(self.nb_layers - 1):
            pred = np.matmul(self.weight_matrices[idx], pred) + self.bias[idx]
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
        #print("predicted", y_hat)
        #print("truth", y)
        if not isinstance(y_hat, np.ndarray) or not isinstance(y, np.ndarray):
            print("model.lossbce : bad argument")
            return None
        if y_hat.shape[0] != y.shape[0]:
            print("model.lossbce : bad dimensions")
            return None
        ret = 0
        for idx in range(y.shape[0]):
            ret = ret + (y[idx][0] * math.log(y_hat[idx][0] + 1e-15) + (1 - y[idx][0])* math.log(1 - y_hat[idx][0] + 1e-15))
            ret = ret + (y[idx][1] * math.log(y_hat[idx][1] + 1e-15) + (1 - y[idx][1])* math.log(1 - y_hat[idx][1] + 1e-15))
        return (- ret / (2 * y.shape[0]))

    def fit(self, network, data_train, data_valid, truth, truthv, loss='binaryCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84):
        if (truth.shape[0] != data_train.shape[0]):
            print("model:fit Dimension error")
        print("Before training weights", self.weight_matrices)
        print("Before training biais", self.bias)
        elem = 0
        listloss = []
        for steps in range(epochs):
            (minigradw, minigradb) = self.__create_minigrad()
            #print("Minigrad weight ini", minigradw) working
            #print("Minigrad bias ini", minigradb) working

            for i in range(batch_size):
                #print("i", (elem + i) % data_train.shape[0])
                neurons = self.__forwarding(data_train[(elem + i) % data_train.shape[0]])
                #print("neurons", neurons)
                for layerid in range(self.nb_layers - 2, -1, -1):
                    #weightL = self.weight_matrices[layerid]
                    #biasm = np.zeros(self.bias[layerid].shape)
                    #grad_last = np.zeros(weightL.shape)
                    #print("layerid", layerid)

                    neuronsLast = neurons[self.nb_layers - 1]
                    neuronsL = neurons[layerid + 1]
                    if (layerid == self.nb_layers - 2):
                        diff = np.array(neuronsLast - truth[(elem + i) % data_train.shape[0]])
                        delta = diff
                        #prev_var = np_array(neuronsLast - truth[i])
                        #biasm = biasm + diff
                    else:
                        diff = np.matmul(delta, self.weight_matrices[layerid + 1])
                        #print("diff", diff)
                        #print("derr", model.__derivative(neuronsL, self.Layers[layerid + 1].activation))
                        diff = diff * model.__derivative(neuronsL, self.Layers[layerid + 1].activation)
                        #print("res", diff)
                        delta = diff

                        #biasm = biasm + diff
                    #if (layerid == self.nb_layers -2):
                    #print("ok")
                    #print("delta", delta)
                    #print("diff", diff)
                    minigradb[layerid] = minigradb[layerid] + diff
    
                    diff = diff.reshape(1, -1)
                    neuronsL1 = np.array(neurons[layerid]).reshape(-1, 1)
                    grad = np.matmul(neuronsL1, diff).transpose()
                    #if (layerid == self.nb_layers - 2):
                    #print("ok2")
                    minigradw[layerid] = minigradw[layerid] + grad
                    #print("hello", layerid)
                    #grad_last = grad_last + grad

            #grad_last = grad_last / data_train.shape[0]
            #biasm = biasm / data_train.shape[0]

            #print('minigradb', minigradb)
            #print('minigradw', minigradw)

            elem = elem + batch_size

            for idx in range(self.nb_layers - 1):
                self.weight_matrices[idx] = self.weight_matrices[idx] - (learning_rate * (minigradw[idx] / batch_size))
            for idx in range(self.nb_layers - 1):
                self.bias[idx] = self.bias[idx] - (learning_rate * (minigradb[idx] / batch_size))
            #print('result', self.weight_matrices)

            print("wait", self.weight_matrices[0])
            #print("biais", self.bias[0])

            #self.weight_matrices[self.nb_layers - 2] = weightL - (learning_rate * grad_last)
            #self.bias[self.nb_layers - 2] = self.bias[self.nb_layers - 2] - (learning_rate * biasm)
            Y_hat = self.predict(data_train)
            Y_vhat = self.predict(data_valid)

            loss = self.lossbce(Y_hat, truth)
            val_los = self.lossbce(Y_vhat, truthv)

            listloss.append((loss, val_los))


            print("epoch {}/{} - loss: {} - val_los : {}".format(steps, epochs, loss, val_los))

            # for upd in range(self.nb_layers - 1):
            #     (x, y) = (self.weight_matrices[upd].shape[0], self.weight_matrices[upd].shape[1])
            #     self.weight_matrices[upd] = sqrt(2.0 / x) * np.random.randn(x, y)

            #for k in range(last_layer.shape[0]):


            #print(last_layer.values[0], last_layer.values[1])
        x_epoch = np.arange(0, epochs)
        listloss = np.array(listloss)

        plt.plot(x_epoch, listloss[:, 0], label='training loss')
        plt.plot(x_epoch, listloss[:, 1], '--', label='validation loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

        # plt.plot(x_epoch, listloss[:2])
                


