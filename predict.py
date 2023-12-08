import pickle
import pandas as pd
import numpy as np
import argparse
from metrics import *
from model import *

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


def parse_arguments () -> tuple:
    try:
        parser = argparse.ArgumentParser(
            prog='predict.py',
            description="A program that gives the prediction of a dataset through the neural network model saved in saved_model.npy and evaluate its performance."
        )
        parser.add_argument(
            "-d", "--dataset",
            dest="dataset_path",
            type=str,
            help="path to the testing dataset.",
            default="data_test.csv"
        )
        args = parser.parse_args()
        return (args.dataset_path)
    except Exception as e:
        print("Error parsing arguments: ", e)
        exit(1)

if __name__ == '__main__':

    try:
        file = open('./saved_model.npy', 'rb')
        mynetwork = pickle.load(file)
        file.close()
    except:
        print("Error loading model")
        exit(1)

    if not isinstance(mynetwork, model):
        print("Error loading model")
        exit(1)

    d_path = parse_arguments()

    test = load_csv(d_path)
    train = load_csv("data_train.csv")

    if (test is None or train is None):
        print("Error reading datasets")
        exit(1)

    X_ref = train.loc[:, 2:].to_numpy()

    X_test = test.loc[:, 2:].to_numpy()
    X_test = normalize_data(X_test, X_ref)
    Y_test = test.loc[:, 1].to_numpy()

    Y_test = np.array([[1, 0] if i == 'M' else [0, 1] for i in Y_test])

    pred_test = mynetwork.predict(X_test)

    rounded = [[round(p[0], 3), round(p[1], 3)] for p in pred_test]
    lossbce = mynetwork.lossbce(pred_test, Y_test)
    lossmse = mynetwork.lossmse(pred_test, Y_test)
    (prec, reca, f1) = getmetrics(Y_test, pred_test)

    truth = test.loc[:, 1].to_numpy()
    truth = np.array([1 if i == 'M' else 0 for i in truth])
    pred = [1 if p[0] > p[1] else 0 for p in pred_test]
    correct = 0
    for i in range(pred_test.shape[0]):
        print("->({}, {}) - raw {}".format(truth[i], pred[i], rounded[i]))
        if (pred[i] == truth[i]):
            correct=correct + 1
    print("> correctly predicted {}/{}".format(correct, pred_test.shape[0]))
    print("> performance :", correct / pred_test.shape[0] * 100, "%")
    print("> loss (binary crossentropy) :", lossbce)
    print("> loss (mean square error) :", lossmse)
    print("> precision :", prec)
    print("> recall :", reca)
    print("> f1_score :", f1)