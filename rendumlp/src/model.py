from math import sqrt
from math import log
from numpy.random import randn
import numpy as np
from src.utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.metrics import *
import pickle

class model():
    def __init__(self, nb_layers , weight_matrices, bias, Layers):
        self.nb_layers = nb_layers
        self.weight_matrices = weight_matrices
        self.Layers = Layers
        self.bias = bias

    def createNetwork(listLayers):
        weight_matrices = []
        bias = []

        weight_matrices.append(model.__initialize_weight(listLayers[0].nb_neurons, listLayers[0].nb_neurons, listLayers[0].weights_initializer))
        bias.append(np.zeros(listLayers[0].nb_neurons))
        for idx in range (len(listLayers) - 1):
            weight_matrices.append(model.__initialize_weight(listLayers[idx+1].nb_neurons, listLayers[idx].nb_neurons, listLayers[idx+1].weights_initializer))
            bias.append(np.zeros(listLayers[idx+1].nb_neurons))

        return (model(len(listLayers), weight_matrices, bias, listLayers))

    def __create_minigrad(self):
        grad_weight = []
        grad_bias = []
        grad_bias.append(np.zeros(self.Layers[0].nb_neurons))
        grad_weight.append(np.zeros((self.Layers[0].nb_neurons, self.Layers[0].nb_neurons)))
        for idx in range(self.nb_layers - 1):
            grad_weight.append(np.zeros((self.Layers[idx + 1].nb_neurons, self.Layers[idx].nb_neurons)))
            grad_bias.append(np.zeros(self.Layers[idx+1].nb_neurons))
        return (grad_weight, grad_bias)

    def __initialize_weight(nb_input, nb_output, weights_initializer):
        np.random.seed(42)
        if weights_initializer == "heUniform":
            res = sqrt(2.0 / nb_input) * np.random.randn(nb_input, nb_output)
        if weights_initializer == "zero":
            return np.full((nb_input, nb_output), 0.0)
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
        if function_type == 'relu':
            return np.array([1 if i > 0 else 0 for i in array])
        if function_type == 'sigmoid':
            return np.array([ i * (1 - i) for i in array])
        if function_type == 'softmax':
            s = np.array([i * (1 - i) for i in array]) * 2
            return s 

    def predict(self, input_data):
        if (not isinstance(input_data, np.ndarray)):
            print("model.predict : bad argument")
            return None
        if input_data.shape[1] != self.Layers[0].nb_neurons:
            print("model:predict wrong dimensions")
            return None
        res = []
        for i in range(input_data.shape[0]):
            pred = input_data[i]
            for idx in range(self.nb_layers):
                pred = np.matmul(self.weight_matrices[idx], pred) + self.bias[idx]
                pred = model.__activation(pred, self.Layers[idx].activation)
            res.append(pred)
        return np.array(res)

    def forwarding(self, load):
        res = []
        if not isinstance(load, np.ndarray):
            print("model:forwarding type error")
        if len(load) != self.Layers[0].nb_neurons:
            print("model:forwarding wrong dimensions")
            return None
        pred = load
        for idx in range(self.nb_layers):
            pred = np.matmul(self.weight_matrices[idx], pred) + self.bias[idx]
            pred = model.__activation(pred, self.Layers[idx].activation)
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
            print("model.lossbce : bad dimensions")
            return None
        ret = 0
        for idx in range(y.shape[0]):
            ret = ret + (y[idx][0] * math.log(y_hat[idx][0] + 1e-15) + (1 - y[idx][0])* math.log(1 - y_hat[idx][0] + 1e-15))
        return (- ret / (y.shape[0]))

    def lossce(self, y_hat, y):
        if not isinstance(y_hat, np.ndarray) or not isinstance(y, np.ndarray):
            print("model.lossce : bad argument")
            return None
        if (y_hat.shape[0] != y.shape[0]):
            print("model.lossce : bad dimensions")
            return None
        vlog= np.vectorize(math.log)
        ret = 0
        for idx in range(y.shape[0]):
            ret = ret + np.sum(y[idx] * vlog(y_hat[idx]))
        return (-ret / (y.shape[0]))

    def fit(self, network, data_train, data_valid, truth, truthv, loss='binaryCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84, momentum=0, stop=False, histo=False):
        if (truth.shape[0] != data_train.shape[0] or truthv.shape[0] != data_valid.shape[0]):
            print("model:fit Dimension error")
        elem = 0
        listloss = []
        (chargew, chargeb) = self.__create_minigrad()
        for steps in range(epochs):
            (minigradw, minigradb) = self.__create_minigrad()

            for i in range(batch_size):
                input_neuron =  data_train[(elem + i) % data_train.shape[0]]
                neurons = self.forwarding(input_neuron)
                for layerid in range(self.nb_layers - 1, -1, -1):

                    neuronsLast = neurons[-1]
                    if layerid == 0:
                        neuronsL = neurons[0]
                    else:
                        neuronsL = neurons[layerid - 1]
                    if (layerid == self.nb_layers - 1):
                        y = truth[(elem + i) % data_train.shape[0]]
                        diff = np.array(neuronsLast - y)
                        diff = -(y / neuronsLast - (1 - y) / (1 - neuronsLast)) / y.shape[0]
                        delta = model.__derivative(neuronsLast, self.Layers[-1].activation) * diff
                        delta = np.array([delta])
                    else:
                        delta = np.matmul(delta, self.weight_matrices[layerid + 1])
                        delta = model.__derivative(neurons[layerid], self.Layers[layerid].activation) * delta
                    minigradb[layerid] = minigradb[layerid] + delta[0]
                    if layerid == 0:
                        neuronsL = input_neuron
                    neuronsL = np.array([neuronsL]).reshape(-1, 1)
                    minigradw[layerid] = minigradw[layerid] + np.matmul(neuronsL, delta).T
                    

            elem = elem + batch_size

            for idx in range(self.nb_layers):
                chargew[idx] = (momentum * chargew[idx]) - (learning_rate * (minigradw[idx] / batch_size))
                chargeb[idx] = (momentum * chargeb[idx]) - (learning_rate * (minigradb[idx] / batch_size)) 
            for idx in range(self.nb_layers):
                self.weight_matrices[idx] = self.weight_matrices[idx] + chargew[idx]
                self.bias[idx] = self.bias[idx] + chargeb[idx]
            
            Y_hat = self.predict(data_train)
            Y_vhat = self.predict(data_valid)
            loss = self.lossce(Y_hat, truth)
            val_los = self.lossce(Y_vhat, truthv)

            (prec, rec, f1) = getmetrics(truthv, Y_vhat)

            if (stop is True and steps > 10 and steps > epochs / 10 and val_los > listloss[-1][1]):
                print("Overfitting detected, early stopping at step {}/{}.".format(steps, epochs))
                epochs = steps
                break
            listloss.append((loss, val_los, prec, rec, f1))


            (precr, recr, f1r) = (round(prec, 3), round(rec, 3), round(f1, 3))
            print("epoch {}/{} - loss: {} - val_los : {} - precision : {} - recall {} - f1_score {}".format(steps+1, epochs, loss, val_los, precr, recr, f1r))


        listloss = np.array(listloss)

        if (histo):
            file = open("historic_metrics.pkl", "ab")
            pickle.dump([listloss], file)
            file.close()
        
            datahisto = []
            with open("historic_metrics.pkl", 'rb') as fi:
                try:
                    while True:
                        datahisto.append(pickle.load(fi))
                except EOFError:
                    fi.close()
                    pass
            for nb_met in range(len(datahisto)):
                nummodel = 'model ' + str(nb_met)
                x_epoch = np.arange(0, datahisto[nb_met][0].shape[0])
                plt.plot(x_epoch, datahisto[nb_met][0][:, 1], label=nummodel)

            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.legend()
            plt.show()
            return self

        x_epoch = np.arange(0, epochs)
        plt.plot(x_epoch, listloss[:, 0], label='training loss')
        plt.plot(x_epoch, listloss[:, 1], '--', label='validation loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
        return self