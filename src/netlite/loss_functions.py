import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    '''Interface for loss functions'''
    def __init__(self):
        self.use_logits = False    

    @abstractmethod
    def forward(self, X):
        '''Propagate input forward through the loss.'''
        pass

    @abstractmethod
    def backward(self, grad_backward):
        '''Propagate output through the loss.'''
        pass
    
class MseLoss(LossFunction):
    '''Mean squared error loss'''
    def forward(self, y_model, y_true):
        loss = 0.5 * (y_model-y_true)**2
        return loss.sum()

    def backward(self, y_model, y_true):
        return y_model - y_true

class HuberLoss(LossFunction):
    '''Huber loss function
       - Quadratic loss (L2 norm) for small errors |z| < delta,
         where delta is approx. the data standard deviation.
       - Linear loss (L1 norm) beyond for robustness against outliers.
    '''
    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        
    def forward(self, y_model, y_true):
        z = y_model - y_true

        idx_quadratic = abs(z) < self.delta
        idx_linear    = np.logical_not(idx_quadratic)
        
        loss = np.zeros(z.shape)
        loss[idx_quadratic] = 0.5*z[idx_quadratic]**2
        loss[idx_linear]    = self.delta * (abs(z[idx_linear]) - 0.5*self.delta)
        
        return loss.sum()

    def backward(self, y_model, y_true):
        z = y_model - y_true

        idx_quadratic  = abs(z) < self.delta
        idx_linear_pos = np.logical_and(np.logical_not(idx_quadratic), z > 0)
        idx_linear_neg = np.logical_and(np.logical_not(idx_quadratic), z < 0)

        grad = np.zeros(z.shape)
        grad[idx_quadratic]  = z[idx_quadratic]
        grad[idx_linear_pos] = self.delta
        grad[idx_linear_neg] = -self.delta

        return grad

class CrossEntropyLoss(LossFunction):
    ''' Multi-class cross-entroy loss
        for training of a softmax classifier.
    '''
    def __init__(self):
        '''The cross-entropy loss is trained using logits as the model output,
           i.e. the softmax probabilities are not computed during training.'''
        self.use_logits = True

    def forward(self, logits, y_true):
        # input:  model outputs (logits) and vector of true class indices
        # output: softmax cross-entropy loss
        assert np.issubdtype(y_true.dtype, np.integer), f"Expected integer dtype but got {y_true.dtype}"
        
        batch_size = len(logits)
        assert len(y_true.shape) == 1, "Vector of true class indices should be 1-dimensional."
        assert y_true.shape[0] == batch_size, "Expected exactly one true class per sample."

        true_class_logits = logits[np.arange(batch_size), y_true]
        
        cross_entropy = - true_class_logits + np.log(np.sum(np.exp(logits), axis=-1))
        return cross_entropy.sum()

    def backward(self, logits, y_true):
        # convert to one-hot-encoding:
        ones_true_class = np.zeros_like(logits)
        ones_true_class[np.arange(len(logits)),y_true] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        
        return -ones_true_class + softmax
