import pandas as pd
import argparse
import numpy as np
import sys
from model import model
from layers import layers
from utils import *

def parse_arguments () -> tuple:
    try:
        parser = argparse.ArgumentParser(
            prog='train.py',
            description="A program that permorf a training on data_train.csv with multilayer-perceptron"

        )

        parser.add_argument("--layer", nargs='+', help="number of neurons of the hidden layer", default=[24, 24, 24])
        parser.add_argument("--epochs", type=int, help="number of iteration of the gradient descent algorythm", default=84)
        parser.add_argument("--loss", type=str, help="Type of loss used to determine the error from the model to fit the data", default="binaryCrossentropy")
        parser.add_argument("--batch-size", type=int, help="Number of elements from the dataset used to compute the gradient on each epochs", default=8)
        parser.add_argument("--learning_rate", type=float, help="fraction of the gradient to update the model", default=0.0314)
        
        args = parser.parse_args()

        return(args.layer, arg)
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

    test = load_csv("data_test.csv")
    train = load_csv("data_train.csv")

    X_ref = train.loc[:, 2:].to_numpy()
    Y_train = train.loc[:, 1].to_numpy()
    Y_train = np.array([[1, 0] if i =='M' else [0, 1] for i in Y_train])

    X_test = test.loc[:, 2:].to_numpy()
    Y_test = test.loc[:, 1].to_numpy()
    Y_test = np.array([[1, 0] if i == 'M' else [0, 1] for i in Y_test])

    X_train = normalize_data(X_ref, X_ref)
    X_test = normalize_data(X_test, X_ref)




    mynetwork = model.createNetwork([
        layers.DenseLayer(X_train.shape[1], activation='sigmoid'),
        layers.DenseLayer(24, activation='sigmoid', weights_initializer="heUniform"),
        layers.DenseLayer(24, activation='sigmoid', weights_initializer="heUniform"),
        layers.DenseLayer(2, activation='softmax', weights_initializer= "heUniform")
    ])

    mynetwork.fit(mynetwork, X_train, X_test, Y_train, Y_test, epochs=2000, learning_rate=0.7, batch_size=X_train.shape[0])

    pred_test = mynetwork.predict(X_test)
    pred_train = mynetwork.predict(X_train)
    # print(pred_test)
    # for loop in range(pred_test.shape[0]):
    #     if (pred_test[loop][0] > pred_test[loop][1]):
    #         print("[1, 0] ->", Y_test[loop])
    #     else:
    #         print("[0, 1] ->", Y_test[loop])
    
    truth = train.loc[:, 1].to_numpy()
    truth = np.array([1 if i == 'M' else 0 for i in truth])
    pred = [1 if i[0] > i[1] else 0 for i in pred_train]
    correct = 0
    for i in range(pred_train.shape[0]):
        #print("->({}, {}) - raw {}".format(truth[i], pred[i], pred_test[i]))
        if (pred[i] == truth[i]):
            correct = correct + 1
    train_ratio = correct / pred_train.shape[0] * 100



    truth = test.loc[:, 1].to_numpy()
    truth = np.array([1 if i == 'M' else 0 for i in truth])
    pred = [1 if i[0] > i[1] else 0 for i in pred_test]
    correct = 0
    for i in range(pred_test.shape[0]):
        print("->({}, {}) - raw {}".format(truth[i], pred[i], pred_test[i]))
        if (pred[i] == truth[i]):
            correct=correct + 1
    print("Total result : ", correct / pred_test.shape[0] * 100, "%")
    print(train_ratio, "on trained data")
