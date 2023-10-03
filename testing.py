from model import model
from layers import layers
from utils import *


def normalize_data(array, train):
    res = np.copy(array).astype("float")

    for i in range(array.shape[1]):
        max_v = np.max(train[:, i])
        min_v = np.min(train[:, i])
        for j in range(array.shape[0]):
            res[j, i] = (res[j, i] - min_v) / (max_v - min_v)
    return res

if __name__ == "__main__":

    #mylayer = layers.DenseLayer(12, activation='sigmoid', weights_initializer='heUniform')

    mynetwork = model.createNetwork([
        layers.DenseLayer(4, activation='sigmoid'),
        layers.DenseLayer(2, activation='sigmoid', weights_initializer="zero"),
        layers.DenseLayer(2, activation='softmax', weights_initializer="zero")
    ])
    X_train = np.array([[18, 20, 7, 3], [2, 6, 15, 12], [17, 15, 8, 5], [3, 8, 19, 16]])
    X_test = np.array([[4, 7, 16, 14], [3, 6, 19, 18], [12, 13, 1, 8]])

    X_train = normalize_data(X_train, X_train)
    X_test = normalize_data(X_test, X_test)
    #res = mynetwork.predict(X_train)
    want = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    wantv = np.array([[0, 1], [0, 1], [1, 0]])

    mynetwork.fit(mynetwork, X_train, X_test, want, wantv, epochs=10, learning_rate=0.1, batch_size=4)

    # pred = mynetwork.predict(X_train)
    # for i in range(pred.shape[0]):
    #     if pred[i][0] > pred[i][1]:
    #         print("1-0", want[i], "raw {}".format(pred[i]))
    #     else:
    #             print("0-1", want[i], "raw {}".format(pred[i]))        



    # print("predictions", res)
    # print("lossmse", mynetwork.lossmse(res, want))
    # print("lossbce", mynetwork.lossbce(res, want))

    # print("nb layers", mynetwork.nb_layers)
    # print("weight matrices", mynetwork.weight_matrices[0])
    # print("weight matrices", mynetwork.weight_matrices[1])