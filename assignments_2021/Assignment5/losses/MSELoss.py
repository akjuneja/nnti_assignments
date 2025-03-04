import numpy as np

class MSELoss:
    def __init__(self) -> None:
        pass

    def __call__(self, y_true, y_pred):
        loss = (sum((y_true - y_pred)**2) / len(y_true))
        return loss
    
    def backward(self):
        pass