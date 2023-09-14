import math
import numpy as np

def sigmoid(vec):
    if type(vec) is int:
        return 1 / (1 + math.exp(-vec))
    return np.array([1 / (1 + math.exp(-i))  for i in vec])

def softmax(vec):
    norm = np.sum(np.array([math.exp(i) for i in vec]))
    return np.array([math.exp(i) / norm for i in vec])