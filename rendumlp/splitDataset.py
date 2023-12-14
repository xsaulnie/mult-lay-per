import pandas as pd
import argparse
import numpy as np
import sys

def parse_arguments() -> tuple:
    try:
        parser = argparse.ArgumentParser(
            prog="splitDataset.py",
            description="A program designed to split a csv dataset into a csv training dataset and a csv test dataset."
        )

        parser.add_argument('--file', '-f', type=str, help="path to the dataset.", default="./data/data.csv")
        parser.add_argument('--ratio', '-r', dest='ratio', type=int, help='ratio in percent of the training dataset.', nargs='?', default=50)
        args = parser.parse_args()

        if (args.ratio >= 100 or args.ratio <= 0):
            print("The ratio must be beetwin 0 and 100 strictly")
            exit(0)
        return(args.file, args.ratio)
    except Exception as e:
        print("Error parsing arguments: ", e)
        exit(0)

def load_csv(name):

    try:
        df =  pd.read_csv(name, header=None)
    except:
        print("Error loading csv")
        return None
    print("'" + name + "'", "dataset of size", df.shape, "loaded.")
    return df

if __name__ == '__main__':

    (dataset_path, ratio) = parse_arguments()
    
    df = load_csv(dataset_path)

    print("Splitting", dataset_path, "with ratio", ratio)

    elements = int(df.shape[0] * ratio / 100)
    df[0:elements].to_csv('data/data_train.csv', index=False, header=None)
    print("data_train.csv of size", df[0:elements].shape[0], "generated.")
    df[elements:].to_csv('data/data_test.csv', index=False, header=None)
    print("data_test.csv of size", df[elements:].shape[0], "generated.")
    exit(0)
