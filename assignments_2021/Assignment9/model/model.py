import activations
import layers
import numpy as np

class Model:
    def __init__(self, components) -> None:
        '''
        expects a list of components of the model in order with which they must be applied
        '''
        self.components = components
        self.v_initialized = False
        self.sq_grad_sums_initialized = False
        self.adam_moments_initialized = False

    def forward(self, x):
        '''
        performs forward pass on the input x using all components from self.components
        '''
        for component in self.components:
            x = component(x)
        return x
        
    def backward(self, in_grad):
        '''
        expects in_grad - a gradient of the loss w.r.t. output of the model
        in_grad must be of the same size as the output of the model

        returns dictionary, where 
            key - index of the component in the component list
            value - value of the gradient for that component
        '''
        num_components = len(self.components)
        grads = {}
        for i in range(num_components-1, -1, -1):
            component = self.components[i]
            if component.get_type() == 'activation':
                in_grad = component.grad(in_grad)
            elif component.get_type() == 'layer':
                weights_grad, in_grad = component.grad(in_grad)
                grads[i] = weights_grad
            else:
                raise Exception
        return grads

    def update_parameters(self, grads, lr):
        '''
        performs one gradient step with learning rate lr for all components
        ''' 
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            self.components[i].weights = self.components[i].weights - lr * grad

    def sgd_momentum(self, grads, lr, alpha):
        if not self.v_initialized:
            self.v = {}
        
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            if not self.v_initialized:
                self.v[i] = np.zeros_like(grad)
            self.v[i] = alpha * self.v[i] - lr * grad
            self.components[i].weights = self.components[i].weights + self.v[i]
        self.v_initialized = True
    
    def ada_grad(self, grads, lr, tol=1e-8):
        if not self.sq_grad_sums_initialized:
            self.sq_grad_sums = {}
        
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            if not self.sq_grad_sums_initialized:
                self.sq_grad_sums[i] = np.zeros_like(grad)
            self.sq_grad_sums[i] += np.power(grad, 2)
            delta = -np.multiply(lr / (tol + np.sqrt(self.sq_grad_sums[i])), grad)
            self.components[i].weights = self.components[i].weights + delta
        self.sq_grad_sums_initialized = True
    
    def adam(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-08):
        if not self.adam_moments_initialized:
            self.adam_moments = {}
        for i, grad in grads.items():
            assert self.components[i].get_type() == 'layer'
            if not self.adam_moments_initialized:
                self.adam_moments[i] = (np.zeros_like(grad), np.zeros_like(grad),)
                self.epoch = 0
            self.epoch += 1
            
            biased_moments = (
                beta1*self.adam_moments[i][0] + (1-beta1)*grad, 
                beta2*self.adam_moments[i][1] + (1-beta2)*np.multiply(grad, grad),
            )
            self.adam_moments[i] = biased_moments

            first_moment = self.adam_moments[i][0] / (1 - beta1**self.epoch)
            second_moment = self.adam_moments[i][1] / (1 - beta2**self.epoch)
            self.components[i].weights = self.components[i].weights - lr*first_moment / (np.sqrt(second_moment) + eps)
        self.adam_moments_initialized = True