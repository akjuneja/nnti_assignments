import numpy as np
import copy

class Linear:
    def __init__(self, in_features, out_features, bias = True):
        self.weights = np.random.randn(in_features, out_features) * 0.05
        self.bias    = np.random.randn(1, out_features) * 0.05


    def __call__(self, x):
        out = x @ self.weights + self.bias
        # save the current input (should be a minibatch)
        self.X = x
        return out

    def get_type(self):
        return 'layer'

    def grad(self, in_gradient):
        '''
        expects in_gradient of size minibatch_size, out_features
        returns dL/dW (size equal to the size of weight matrix) 
                dL/dX (size equal to the size of input matrix)
        '''
        x_transpose = self.X.transpose()
        w_transpose = self.weights.transpose()
        print(in_gradient.shape)
        print(w_transpose.shape)
        g_x = np.matmul(in_gradient, w_transpose)
        g_w = np.matmul(x_transpose, in_gradient)
        return g_w, g_x

