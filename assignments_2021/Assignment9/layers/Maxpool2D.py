import numpy as np

class Maxpool2D:
    def __init__(self, kernel = 2, stride = 1):
        self.kernel = kernel # Size of the kernel. We assume that the kernel has same height and width
        self.stride = stride # Stride of the pooling operation
        self.X = None       # Used to store the value of X for backpropagation

    def __call__(self, X):
        """
        Implements the forward pass of the pooling layer
        
        Arguments:
        X: numpy array of shape (batch_size, n_H_in, n_W_in, n_C_in)
        
        Returns:
        A: output of the pool layer, a numpy array of shape (batch_size, n_H, n_W, n_C)
        """
        # Store the input for backpropagation
        self.X = X

        # Retrieve dimensions from the input shape
        batch_size, n_H_in, n_W_in, n_C_in = X.shape
        
        
        # Define output dimensions
        n_H_out = int(1 + (n_H_in - self.kernel) / self.stride)
        n_W_out = int(1 + (n_W_in - self.kernel) / self.stride)
        n_C_out = n_C_in
        
        # Output matrix A is initilized to 0 and you have to return this matrix A
        # After applying the max pooling operations on X
        A = np.zeros((batch_size, n_H_out, n_W_out, n_C_out))              
        

        ########################################################################
        # TODO: Implement the following pseudo-code
        # loop over batch_size:
        #   loop over the vertical axis of A:
        #       loop over horizontal axis of A:
        #           loop over output channels of A:
        #               Find the indexes of window in input X
        #               Apply max pooling operation over the window
        #               Assign the max value to the correct index in A 
        ########################################################################
        
    
        return A
    
    def get_type():
        return 'layer'

    def grad(self, in_gradient):
        """
        Implements the backward pass of the pooling layer
        
        Arguments:
        in_gradient: gradient of cost with respect to the output of the pooling layer
                    same shape as A in the `__call__()`
        
        Returns:
        dLdX: gradient of the cost w.r.t to the input, 
                numpy array with the same shape as X
        """

        # Retrieve dimensions from Self.x shape and in_gradient shape
        batch_size, n_H_in, n_W_in, n_C_in  = self.X.shape
        batch_size, n_H, n_W, n_C           = in_gradient.shape
        
        # Initialize dLdX
        dLdX = np.zeros(self.X.shape)
        
        ########################################################################
        # TODO: Implement the following pseudo code
        # loop over batch_size:
        #   loop over the vertical axis of DLdX:
        #       loop over horizontal axis of DLdX:
        #           loop over output channels of DLdX:
        #               Find the indexes of window in input X
        #               Extract the window from X using the indices
        #               Create a mask of the window using `self.create_mask` function
        #               Apply the same mask to in_gradient and store the value in DLdX
        ########################################################################
        
        return dLdX, None

    def create_mask(self, X):
        # TODO: SUPPLY as described in the exercise sheet
        return None