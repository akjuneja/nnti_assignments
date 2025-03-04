import numpy as np

class CrossEntropy:
    def __init__(self, class_count=None, average=True):
        self._EPS = 1e-8
        self.classes_counts = class_count
        self.average = average
        
    def __call__(self, Y_pred, Y_real):
        '''
        expects: Y_pred - N*D matrix of predictions (N - number of datapoints)
                 Y_real - N*D matrix of one-hot vectors 
        applies softmax before computing negative log likelihood loss
        return a scalar
        '''
        self.y = Y_pred
        Y_pred = Y_pred - np.max(Y_pred, axis=1)[:, None]
        self.y_pred = Y_pred
        self.y_real = Y_real
        self.N = Y_pred.shape[0]

        # applying softmax function 
        self.probabilities = np.exp(Y_pred[Y_real.astype(bool)]) / np.sum(np.exp(Y_pred), axis=1)
        logs = np.log(self.probabilities+self._EPS)

        if self.average:
            return -np.sum(logs) / float(self.N)
        return -np.sum(logs)

    def grad(self):
        '''
        returns gradient with the size equal to the the size of the input vector (self.y_pred)
        '''
        exp_s = np.sum(np.exp(self.y), axis=1)
        exp_row  = np.exp(self.y)
        for i in range(exp_s.shape[0]):
            exp_row[i] = np.divide(exp_row[i], exp_s[i])
        g = exp_row - self.y_real
        return g/self.N