import pandas as pd
import argparse
import numpy as np
from src.model import model
from src.layers import layers
from src.utils import *
import pickle

def parse_arguments () -> tuple:
    try:
        parser = argparse.ArgumentParser(
            prog='train.py',
            description="A program that perform a training on data_train.csv with multilayer-perceptron"

        )
        parser.add_argument("--datasetrain",type=str, help="dataset for the training of the neural network", default='data/data_train.csv')
        parser.add_argument("--datasetest", type=str, help="dataset for the testing of the trained neural network", default='data/data_test.csv')
        parser.add_argument("-l", "--hidenlayer", type=int, nargs='+', help="number of neurons of the hidden layer", default=[24, 24])
        parser.add_argument("-e", "--epochs", type=int, help="number of iteration of the gradient descent algorythm", default=100)
        parser.add_argument("-b", "--batchsize", type=int, help="number of elements from the dataset used to compute the gradient on each epochs", default=0)
        parser.add_argument("-r", "--learningrate", type=float, help="fraction of the gradient to update the model", default=0.7)
        parser.add_argument("-m", "--momentum", type=float, help="hyperparameter of the nesterov momentum", default=0.5)
        parser.add_argument("-s", "--stop", action='store_true', help='flag that set early stopping to prevent the overfitting')
        parser.add_argument("-d", "--history", action='store_true', help='display several learging curve on same graph using historic of trainings')
        
        args = parser.parse_args()

        if args.momentum > 1 or args.momentum < 0:
            print("Momentum hyperparameter must be beetween 0 and 1")
            exit(1)
        if (len(args.hidenlayer) > 10):
            print("Error Neural network too deep")
            exit(1)
        for i in args.hidenlayer:
            if i > 100:
                print("Error too many neurons on layer")
                exit(1)
        if args.batchsize < 0:
            print("Error batch size must be positiv")
            exit(1)
        if args.learningrate < 0 or args.learningrate > 1:
            print("Error learning rate must be beetweem 0 and 1")
            exit(1)


        return(args.datasetrain, args.datasetest, args.hidenlayer, args.epochs, args.batchsize, args.learningrate, args.momentum, args.stop, args.history)
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

    # best training parameter : python train.py -l 28 26 -e 1200 -m 0.9 -r 0.4 

    (datasetrain, datasetest, hidden_layer, epochs, batch_size, learning_rate, momentum, stop, histo) = parse_arguments()

    test = load_csv(datasetest)
    train = load_csv(datasetrain)

    if (test is None or train is None):
        print("Error loading csv.")
        exit(1)

    if (test is None or train is None):
        print("Error reading datasets")
        exit(1)

    X_ref = train.loc[:, 2:].to_numpy()
    if (batch_size > X_ref.shape[0]):
        print("Error : batch size greater than number of elements in the training dataset")
        exit(1)

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

    mynetwork.fit(mynetwork, X_train, X_test, Y_train, Y_test, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, momentum=momentum, stop=stop, histo=histo)
    print("> saving model './saved_model.npy' to disk...")

    file = open('./saved_model.pkl', 'wb')

    pickle.dump(mynetwork, file)
    file.close()
    exit(0)


