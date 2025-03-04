import numpy as np

class Tanh:
    def __init__(self):
        pass
    
    def __call__(self, x):
        self.x = x
        return np.tanh(x)

    def get_type(self):
        return 'activation'

    def grad(self, in_gradient):
        t = 1 - (np.tanh(self.x) * np.tanh(self.x))
        out = t * in_gradient
        return t