from src.model import model
from src.layers import layers
import numpy as np
if __name__ == "__main__":

    X_train = np.array([[3, 4, 5], [6, 7, 8]])

    Y_train = np.array([[0., 1., 0.], [1., 0., 0.]])

    mod = model.createNetwork([layers.DenseLayer(3, activation='sigmoid'),
    layers.DenseLayer(2, activation='sigmoid'),
    layers.DenseLayer(2, activation='softmax'),
    ])

    mod.fit(mod, X_train, X_train, Y_train, Y_train, epochs=200, batch_size=2)
    exit(0)

