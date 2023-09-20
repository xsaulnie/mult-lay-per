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
    X_train = np.array([[18, 20, 7, 3], [15, 12, 2, 6]])
    X_test = np.array([[4, 7, 16, 14], [3, 6, 19, 18]])
    res = mynetwork.predict(X_train)
    want = np.array([[0, 1], [1, 0]])

    mynetwork.fit(mynetwork, X_train, X_test, want)

    # print("predictions", res)
    # print("lossmse", mynetwork.lossmse(res, want))
    # print("lossbce", mynetwork.lossbce(res, want))

    # print("nb layers", mynetwork.nb_layers)
    # print("weight matrices", mynetwork.weight_matrices[0])
    # print("weight matrices", mynetwork.weight_matrices[1])