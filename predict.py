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
    X_train = np.array([[5, 6, 7, 10], [1, 2, 3, 4]])
    X_test = np.array([[4, 12, 7, 2], [8, 3, 1, 6]])
    res = mynetwork.predict(X_train)
    want = np.array([[0, 1], [1, 0]])
    mynetwork.fit(mynetwork, X_train, X_test)
    # print("predictions", res)
    # print("lossmse", mynetwork.lossmse(res, want))
    # print("lossbce", mynetwork.lossbce(res, want))

    # print("nb layers", mynetwork.nb_layers)
    # print("weight matrices", mynetwork.weight_matrices[0])
    # print("weight matrices", mynetwork.weight_matrices[1])