import numpy as np

class CrossEntropy:
    def __init__(self):
        self._eps = 1e-8 # Add eps to CE loss
        
    def __call__(self, Y_pred, Y_true):
        # Assume $Y_true \in {0,1}$ 
        return(-1 * (Y_true * np.log(Y_pred)).sum())


    def backward(self, Y_pred, Y_true):
        pass