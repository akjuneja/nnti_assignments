import numpy as np
import copy
import torch
import random

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)


class Dropout:
    def __init__(self, layer, p : float = 0.5):
        self.p = p
        self.layer = layer
        self.weights = self.layer.weights
        self.bias = self.layer.bias

    def __call__(self, x):
        '''
            apply inverted dropout. Store the mask that you generate with probability p in
            self.mask
        '''
        nx = self.layer(x)
        bernoulli = torch.distributions.bernoulli.Bernoulli(self.p)
        self.mask = bernoulli.sample(nx.shape) * (1.0/(1-self.p))
        d_x = nx*self.mask.numpy()
        return d_x

    def get_type(self):
        return 'layer'

    def grad(self, in_gradient):
        '''
            Apply the mask to the backward pass of a linear layer.
            The return values () are similar to Linear.py
        '''
        return self.X.T @ self.mask @ in_gradient, in_gradient @ self.weights.T @ self.mask