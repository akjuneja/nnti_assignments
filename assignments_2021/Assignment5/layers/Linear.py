import numpy as np
import copy

class Linear:
    def __init__(self, in_features, out_features, bias = True):
        self.weights = np.random.randn(in_features, out_features)
        self.bias    = np.random.randn(1, out_features)


    def __call__(self, x):
       return (np.matmul(x, self.weights) + self.bias)

    def backward(self, in_gradient):
        pass

