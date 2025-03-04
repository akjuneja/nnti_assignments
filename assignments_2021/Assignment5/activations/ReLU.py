import numpy as np

class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return (np.maximum(0, x))
  
    def backward(self, in_gradient):
        pass