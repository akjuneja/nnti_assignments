import numpy as np

class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, x):
        self.x = x
        return np.multiply(x, (x > 0))

    def get_type(self):
        return 'activation'

    # assign gradient of zero if x = 0 (even though the function is not differentiable at that point)
    def grad(self, in_gradient):
        t = self.x > 0
        t1 = t.astype(int)
        out = t1 * in_gradient
        return out