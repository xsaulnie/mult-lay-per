from model import model
from layers import layers
import numpy as np

network = model.createNetwork([
    layers.DenseLayer(3, activation='sigmoid'),
    layers.DenseLayer(2, activation='sigmoid'),
    layers.DenseLayer(1, activation='sigmoid')
])

#print(network.weight_matrices)

network.weight_matrices[0] = np.array([[0.2, 0.4, -0.5], 
                                       [-0.3, 0.1, 0.2]])
network.weight_matrices[1] = np.array([[-0.3, -0.2]])

#print(network.weight_matrices)

#print(network.bias)

network.bias[0] = np.array([-0.4, 0.2])
network.bias[1] = np.array([0.1])

#print(network.bias)

neurons = network.forwarding(np.array([1, 0, 1]))

print("before", neurons)

X_train = np.array([[1, 0, 1]])
Y_train = np.array([[1]])

X_test = X_train
Y_test = Y_train

network.fit(network, X_train, X_test, Y_train, Y_test, epochs=200, learning_rate=0.1, batch_size=1, loss='other')

neurons = network.forwarding(np.array([1, 0, 1]))

print("after", neurons)