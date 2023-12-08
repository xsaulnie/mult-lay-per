import pandas as pd
import argparse
import numpy as np
from model import model
from layers import layers
from utils import *
import pickle

def parse_arguments () -> tuple:
    try:
        parser = argparse.ArgumentParser(
            prog='train.py',
            description="A program that perform a training on data_train.csv with multilayer-perceptron"

        )
        parser.add_argument("--datasetrain",type=str, help="dataset for the training of the neural network", default='data_train.csv')
        parser.add_argument("--datasetest", type=str, help="dataset for the testing of the trained neural network", default='data_test.csv')
        parser.add_argument("-l", "--hidenlayer", nargs='+', help="number of neurons of the hidden layer", default=[24, 24])
        parser.add_argument("-e", "--epochs", type=int, help="number of iteration of the gradient descent algorythm", default=100)
        parser.add_argument("-b", "--batchsize", type=int, help="number of elements from the dataset used to compute the gradient on each epochs", default=0)
        parser.add_argument("-r", "--learningrate", type=float, help="fraction of the gradient to update the model", default=0.7)
        parser.add_argument("-m", "--momentum", type=float, help="hyperparameter of the nesterov momentum", default=0.5)
        parser.add_argument("-s", "--stop", action='store_true', help='flag that set early stopping to prevent the overfitting')
        
        args = parser.parse_args()

        return(args.datasetrain, args.datasetest, args.hidenlayer, args.epochs, args.batchsize, args.learningrate, args.momentum, args.stop)
    except Exception as e:
        print("Error parsing arguments: ", e)
        exit()

def load_csv(name):

    try:
        df =  pd.read_csv(name, header=None)
    except:
        print("Error loading", name, "as csv")
        return None
    print("'" + name + "'", "dataset of size", df.shape, "loaded.")
    return df

def normalize_data(array, train):
    res = np.copy(array).astype("float")

    for i in range(array.shape[1]):
        max_v = np.max(train[:, i])
        min_v = np.min(train[:, i])
        for j in range(array.shape[0]):
            res[j, i] = (res[j, i] - min_v) / (max_v - min_v)
    return res


if __name__ == '__main__':

    (datasetrain, datasetest, hidden_layer, epochs, batch_size, learning_rate, momentum, stop) = parse_arguments()
    print(datasetrain, datasetest, hidden_layer, epochs, batch_size, learning_rate, momentum, stop)

    test = load_csv(datasetest)
    train = load_csv(datasetrain)

    if (test is None or train is None):
        print("Error reading datasets")
        exit(1)

    X_ref = train.loc[:, 2:].to_numpy()
    Y_train = train.loc[:, 1].to_numpy()
    Y_train = np.array([[1, 0] if i =='M' else [0, 1] for i in Y_train])

    X_test = test.loc[:, 2:].to_numpy()
    Y_test = test.loc[:, 1].to_numpy()
    Y_test = np.array([[1, 0] if i == 'M' else [0, 1] for i in Y_test])

    X_train = normalize_data(X_ref, X_ref)
    X_test = normalize_data(X_test, X_ref)

    listlayers = []
    listlayers.append(layers.DenseLayer(X_train.shape[1], activation='sigmoid'))
    for i in range(len(hidden_layer)):
        listlayers.append(layers.DenseLayer(hidden_layer[i], activation='sigmoid', weights_initializer="heUniform"))
    listlayers.append(layers.DenseLayer(2, activation='softmax', weights_initializer="heUniform"))

    mynetwork = model.createNetwork(listlayers)

    if batch_size == 0:
        batch_size = X_train.shape[0]
    if (batch_size > X_train.shape[0]):
        print("Error batch_size too big, fitting impossible")
        exit(1)

    mynetwork.fit(mynetwork, X_train, X_test, Y_train, Y_test, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, momentum=momentum, stop=stop)
    print("> saving model './saved_model.npy' to disk...")

    file = open('./saved_model.npy', 'wb')

    pickle.dump(mynetwork, file)
    exit(0)


