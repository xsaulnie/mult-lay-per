import pandas as pd
import numpy as np
import sys

def load_csv(name):

    try:
        df =  pd.read_csv(name)
    except:
        print("Error loading csv")
        return None
    print("'" + name + "'", "dataset of size", df.shape, "loaded.")
    return df

if __name__ == '__main__':
    
    df =load_csv(sys.argv[1])