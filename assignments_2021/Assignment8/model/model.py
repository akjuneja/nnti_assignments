import activations
import layers

class Model:
    def __init__(self, components) -> None:
        '''
        expects a list of components of the model in order with which they must be applied
        '''
        self.components = components

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

    def sgd_momentum(self, *args):
        # your implementation of SGD with momentum goes here
        raise NotImplementedError
    
    def ada_grad(self, *args):
        # your implementation of AdaGrad goes here
        raise NotImplementedError
    
    def adam(self, *args):
        # your implementation of Adam goes here
        raise NotImplementedError