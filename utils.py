import math
import numpy as np

def sigmoid(x):
        return 1 / (1 + math.exp(-x))

def softmax(vec):
    norm = np.sum(np.array([math.exp(i[0]) for i in vec]))
    return np.array([math.exp(i[0]) / norm for i in vec]).reshape(-1, 1)