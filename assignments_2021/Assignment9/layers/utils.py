import numpy as np

def zero_padding(X, padding):
    """
    pads images with 0's in a batch
    args:
        X: numpy.ndarray, X.shape = (batch_size, n_H, n_W, n_C)
            Note the difference from pytorch where we use (batch_size, n_C, n_H, n_W)
        padding: integer which tells us how much padding is required on the borders of the image.

    hint: use np.pad()
    
    Returns:
    Xpad: shape = (batch_size, n_H + 2*padding, n_W + 2*padding, n_C)
    """
    ##########################
    Xpad = None # TODO: Supply the code for Xpad
    ##########################
    return Xpad