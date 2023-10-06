from model import model
from layers import layers
import numpy as np

network = model.createNetwork([
    layers.DenseLayer(3, activation='sigmoid'),
    layers.DenseLayer(3, activation='sigmoid'),
    layers.DenseLayer(2, activation='softmax')
])

X_train = np.array([[0.7, 0.4, 0.1]])
Y_train= np.array([[0., 1.]])

X_test = X_train
Y_test = Y_train
print("matrices", network.weight_matrices)
print("bias", network.bias)
network.fit(network, X_train, X_test, Y_train, Y_test, epochs=1, learning_rate=0.1, batch_size=1)
print("matrices", network.weight_matrices)
print("bias", network.bias)