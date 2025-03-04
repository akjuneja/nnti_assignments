import numpy as np

class Tanh:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return(np.sinh(x) / np.cosh(x))
    
    def backward(self):
        pass