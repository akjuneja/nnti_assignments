import numpy as np
from .utils import zero_padding

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=2):
        self.in_channels = in_channels      # The number of channels of the input
        self.out_channels = out_channels    # The number of channels of the output
        self.kernel_size = kernel_size      # Size of the kernel used to convolve over
        self.padding = padding              # Amount of padding to be added 
        self.stride = stride                # Stride of the convolution
        self.X = None                       # Used to store the value of X for backprop

        # Naturally the size of the weights will be k x k x n_C_in x n_C_out
        # Select the weight of the i'th output channel using self.W[:,:,:,i]
        self.W = np.random.randn(kernel_size, kernel_size, self.in_channels, self.out_channels)

        # Size of the bias params will be (1, 1, 1, n_C_out)
        # select the bias of the i'th channel using self.b[:,:,:,i]
        self.b = np.random.randn(1, 1, 1, self.out_channels)

    def __call__(self, X):
        """
        Implements the forward propagation for a convolution function.
        Arguments:
        X: (batch_size, n_H_in, n_W_in, n_C_in)
            
        Returns:
        Z: -- (batch_size, n_H_out, n_W_out, n_C_out)
        cache -- cache of values needed for the conv_backward() function
        """
        # Store X for backpropagation
        self.X = X 

        # Compute dimensions from X shape
        batch_size, n_H_in, n_W_in, n_C_in = X.shape
        
        # Compute the output dimensions
        n_H_out = int((n_H_in - self.kernel_size + 2 * self.padding) / self.stride + 1)
        n_W_out = int((n_W_in - self.kernel_size + 2 * self.padding) / self.stride + 1)
        
        # Initialize the output tensor Z with zeros
        Z = np.zeros((batch_size, n_H_out, n_W_out, self.out_channels))
        
        ########################################################################
        # TODO: Implement the following pseudo code
        # Pad X with zeroes using `zero_padding()` and pad value self.padding
        # Loop over batches of images
        #   Loop over vertical axis of the output
        #       Loop over horizontal axis of output
        #           Loop over the output channels:
        #               find the corresponding indices of the padded input volume
        #               apply convolution of the weights over the window
        #               Store the scalar value of the convolution in Z at the correct index
        ########################################################################
                
        return Z

    def get_type(self):
        return 'layer'

    def grad(self, in_gradient):
        """
        Implement the backward propagation for a convolution function      
        Arguments:
        in_gradient: gradient of the cost with respect to the output of the conv layer
        
        Returns:
        dLdX: gradient of the cost with respect to the input. Same shape as self.X
        dLdW: gradient of the cost with respect to the weights self.W. Same shape as self.W
        dLdb: gradient of the cost with respect to the biases. Same shape as self.b
        """
        
        # Get dimensions from self.X's shape
        batch_size, n_H_in, n_W_in, n_C_in = self.X.shape
        
        # Initialize dLdX, dLdW, dLdb with the correct shapes
        dLdX = np.zeros((batch_size, n_H_in, n_W_in, n_C_in))                           
        dLdW = np.zeros((self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))
        dLdb = np.zeros((1, 1, 1, self.out_channels))

        # Padding X and dLdX with zeros
        Xpadded = zero_padding(self.X, self.padding)
        dLdXpadded = zero_padding(dLdX, self.padding)
        # Use the padded versions in your computations.


        ########################################################################
        # TODO: SUPPLY your code here.
        ########################################################################
        return dLdX, dLdW, dLdb