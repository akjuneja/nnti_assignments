import numpy as np

class LeakyReLU:
    def __init__(self, alpha= 0.01) -> None:
        self.alpha = alpha

    def __call__(self, x):
        data = [max(0.05*val,val) for val in x]
        return np.array(data, dtype=float)
    
    def backward(self):
        pass