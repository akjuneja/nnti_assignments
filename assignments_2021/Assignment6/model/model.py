import activations
import layers
import numpy as np

class Model:
    def __init__(self, components) -> None:
        '''
        expects a list of components of the model in order with which they must be applied
        '''
        self.components = components
        self.components_values = []
        

    def forward(self, x):
        '''
        performs forward pass on the input x using all components from self.components
        '''
        #[layer1, activation1, layer2, activation2]
        
        x = self.components[0](x)
        self.components_values.append(x)
        x = self.components[1](x)
        self.components_values.append(x)
        x = self.components[2](x)
        self.components_values.append(x)
        x = self.components[3](x)
        self.components_values.append(x)
        return x
        
    def backward(self, in_grad):
        '''
        expects in_grad - a gradient of the loss w.r.t. output of the model
        in_grad must be of the same size as the output of the model

        returns dictionary, where 
            key - index of the component in the component list
            value - value of the gradient for that component
        '''
        components_gradients = []
        t = self.components[3].grad(in_grad)
        components_gradients .append(t)
        in_gradient = np.ones((2, 10))
        t, _= self.components[2].grad(in_gradient)
        components_gradients.append(t)
        in_gradient = np.ones((2, 100))
        t = self.components[1].grad(in_gradient)
        components_gradients.append(t)
        t , _= self.components[0].grad(in_gradient)
        components_gradients.append(t)
        return components_gradients 

    def update_parameters(self, grads, lr): 
        '''
        performs one gradient step with learning rate lr for all components
        '''
        self.components[2].weights.data = self.components[2].weights.data - lr*grads[2]
        self.components[0].weights.data = self.components[2].weights.data - lr*grads[0]
        