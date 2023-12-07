from math import sqrt
from math import log
from numpy.random import randn
import numpy as np
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics import *

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
        print(weights_initializer)
        if weights_initializer == "heUniform":
            res = sqrt(2.0 / nb_input) * np.random.randn(nb_input, nb_output)
        if weights_initializer == "zero":
            return np.full((nb_input, nb_output), 0.6)
        #print("init", res)
        return res

    def __activation(array, activation_type):
        if activation_type == 'sigmoid':
            #print("arr", array)
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
        #Waaaarning, must activate
        #pred=load
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
            #ret = ret + (y[idx][1] * math.log(y_hat[idx][1] + 1e-15) + (1 - y[idx][1])* math.log(1 - y_hat[idx][1] + 1e-15))
        return (- ret / (y.shape[0]))

    def fit(self, network, data_train, data_valid, truth, truthv, loss='binaryCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84):
        if (truth.shape[0] != data_train.shape[0] or truthv.shape[0] != data_valid.shape[0]):
            print("model:fit Dimension error")
        #print("Before training weights", self.weight_matrices)
        #print("Before training biais", self.bias)
        elem = 0
        listloss = []
        (chargew, chargeb) = self.__create_minigrad()
        for steps in range(epochs):
            (minigradw, minigradb) = self.__create_minigrad()
            #print("Minigrad weight ini", minigradw)
            #print("comp", self.weight_matrices)
            #print("Minigrad bias ini", minigradb) working

            for i in range(batch_size):
                #print("i", (elem + i) % data_train.shape[0])
                input_neuron =  data_train[(elem + i) % data_train.shape[0]]
                neurons = self.forwarding(input_neuron)
                for layerid in range(self.nb_layers - 1, -1, -1):
                    #print("Layerid", layerid)
                    #weightL = self.weight_matrices[layerid]
                    #biasm = np.zeros(self.bias[layerid].shape)
                    #grad_last = np.zeros(weightL.shape)
                    #print("layerid", layerid)

                    neuronsLast = neurons[-1]
                    if layerid == 0:
                        neuronsL = neurons[0]
                    else:
                        neuronsL = neurons[layerid - 1]
                    if (layerid == self.nb_layers - 1):
                        if loss =='special':
                            diff = neuronsLast * (1 - neuronsLast) * (truth[(elem + i) % data_train.shape[0]] - neuronsLast)
                            ####print("last diff", diff)
                            delta = diff
                        else:
                            y = truth[(elem + i) % data_train.shape[0]]
                            diff = np.array(neuronsLast - truth[(elem + i) % data_train.shape[0]])
                            diff = -(y / neuronsLast - (1 - y) / (1 - neuronsLast)) / 2 #?? / 2
                            #print("first diff", diff, "neuronLast", neuronsLast)
                            #print("derivate", model.__derivative(neuronsLast, "softmax"))
                            delta = model.__derivative(neuronsLast, "softmax") * diff
                            #print("delta", delta)
                            delta = np.array([delta])

                            #exit(0)
                            #print("new input", np.matmul(self.weight_matrices[layerid].T, delta))
                            #exit(0)
                            #prev_var = np_array(neuronsLast - truth[i])
                            #biasm = biasm + diff
                    else:
                        #print("A", self.weight_matrices[layerid + 1])
                        #print("B", delta)
                        delta = np.matmul(delta, self.weight_matrices[layerid + 1])
                        #print("delta, new input", delta)
                        #print("A1", delta)
                        #print("B2", neurons[layerid])
                        delta = model.__derivative(neurons[layerid], self.Layers[layerid].activation) * delta
                        #print("delta", delta)

                        ####print("hiden diff", diff)

                        #biasm = biasm + diff
                    #if (layerid == self.nb_layers -2):
                    #print("ok")
                    #print("delta", delta)
                    #print("diff", diff)
                    #print("TEEEEEEEEEEST", minigradb[layerid])
                    minigradb[layerid] = minigradb[layerid] + delta[0]
                    #print("minigradb", minigradb[layerid])
                    #print("NEUROOOOOONNNNN", delta)
                    if layerid == 0:
                        neuronsL = input_neuron
                    neuronsL = np.array([neuronsL]).reshape(-1, 1)
                    minigradw[layerid] = minigradw[layerid] + np.matmul(neuronsL, delta).T
                    #print("minigradw", minigradw[layerid])
                    


                    #print("hello", layerid)
                    #grad_last = grad_last + grad

            #grad_last = grad_last / data_train.shape[0]
            #biasm = biasm / data_train.shape[0]

            #print('minigradb', minigradb)
            #print('minigradw', minigradw)

            elem = elem + batch_size

            for idx in range(self.nb_layers):
                chargew[idx] = (0.5 * chargew[idx]) - (learning_rate * (minigradw[idx] / batch_size))
                chargeb[idx] = (0.5 * chargeb[idx]) - (learning_rate * (minigradb[idx] / batch_size)) 
            for idx in range(self.nb_layers):
                self.weight_matrices[idx] = self.weight_matrices[idx] + chargew[idx]
                self.bias[idx] = self.bias[idx] + chargeb[idx]

            #print("weight", steps, self.weight_matrices)
            ####print('resultw', self.weight_matrices)
            ####print('resultb', self.bias)

            #print("wait", self.weight_matrices[0])
            #print("biais", self.bias[0])

            #self.weight_matrices[self.nb_layers - 2] = weightL - (learning_rate * grad_last)
            #self.bias[self.nb_layers - 2] = self.bias[self.nb_layers - 2] - (learning_rate * biasm)
            
            Y_hat = self.predict(data_train)
            Y_vhat = self.predict(data_valid)
            loss = self.lossbce(Y_hat, truth)
            val_los = self.lossbce(Y_vhat, truthv)

            (prec, rec, f1) = getmetrics(truthv, Y_vhat) 

            listloss.append((loss, val_los))


            print("epoch {}/{} - loss: {} - val_los : {} - precision : {} - recall {} - f1_score {}".format(steps+1, epochs, loss, val_los, prec, rec, f1))


        x_epoch = np.arange(0, epochs)
        listloss = np.array(listloss)

        plt.plot(x_epoch, listloss[:, 0], label='training loss')
        plt.plot(x_epoch, listloss[:, 1], '--', label='validation loss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

        # plt.plot(x_epoch, listloss[:2])



                    # for upd in range(self.nb_layers - 1):
            #     (x, y) = (self.weight_matrices[upd].shape[0], self.weight_matrices[upd].shape[1])
            #     self.weight_matrices[upd] = sqrt(2.0 / x) * np.random.randn(x, y)

            #for k in range(last_layer.shape[0]):


            #print(last_layer.values[0], last_layer.values[1])