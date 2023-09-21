import math
import numpy as np

def sigmoid(x):
        return 1 / (1 + math.exp(-x))

def softmax(vec):
    norm = np.sum(np.array([math.exp(i) for i in vec]))
    return np.array([math.exp(i) / norm for i in vec])

def relu(vec):
        if isinstance(vec, np.ndarray):
                return np.array([max(0, i) for i in vec])
        return max(0, vec)