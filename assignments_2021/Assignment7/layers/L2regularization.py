import numpy as np
from numpy import linalg as npnorm

class L2regularization(object):
    """
        Implement the class such that it wraps around a linear layer
        and modifies the backward pass of a regularized linear layer
    """

    def __init__(self, layer, coefficient = 0.01):
        """
        """
        assert coefficient > 0., 'Penalty coefficient must be positive.'
        self.coefficient = coefficient
        self.layer = layer
        self.weights = self.layer.weights
        self.bias = self.layer.bias
        
    def __call__(self, x):
        """
        Implements the forward pass of a linear layer
        """
        nx = self.layer(x)
        n = np.square(self.weights)
        n_sum = np.sum(n)
        t = nx + (self.coefficient * n_sum)
        self.x = x
        return t
        
    def grad(self, in_gradient):
        """
        Implements the backward pass of a 'regularized linear layer'
        expects in_gradient of size minibatch_size, out_features
        returns dL/dW (size equal to the size of weight matrix) 
                dL/dX (size equal to the size of input matrix)
        """
        x_transpose = self.x.transpose()
        w_transpose = self.weights.transpose()
        g_x = np.matmul(in_gradient, w_transpose) 
        g_w = np.matmul(x_transpose, in_gradient) + 2*self.coefficient
        return g_w, g_x

    def get_type(self):
        return "layer"