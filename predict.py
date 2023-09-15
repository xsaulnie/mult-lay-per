from model import model
from layers import layers
from utils import *

if __name__ == "__main__":

    mylayer = layers.DenseLayer(12, activation='sigmoid', weights_initializer='heUniform')

    mynetwork = model.createNetwork([
        layers.DenseLayer(4, activation='sigmoid'), 
        layers.DenseLayer(3, activation='sigmoid', weights_initializer="heUniform"),
        layers.DenseLayer(3, activation='sigmoid', weights_initializer="heUniform"),
        layers.DenseLayer(2, activation='softmax', weights_initializer= "heUniform")
    ])

    res = mynetwork.predict(np.array([[5], [6], [7], [10]]))
    print(res)
    print(np.sum(res))

    # print("nb layers", mynetwork.nb_layers)
    # print("weight matrices", mynetwork.weight_matrices[0])
    # print("weight matrices", mynetwork.weight_matrices[1])