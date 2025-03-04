import numpy as np

class Softmax:
    def __init__(self):
        pass
    
    def __call__(self, x):
        e = np.exp(x)
        return e/e.sum()

    def backward(self, in_gradient):
        pass