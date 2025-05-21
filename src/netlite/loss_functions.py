# Copyright (c): Hanno Homann, 2024-'25

import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    '''Interface for loss functions'''
    def __init__(self):
        self.use_logits = False    

    @abstractmethod
    def forward(self, X):
        '''Propagate input forward through the los.'''
        pass

    @abstractmethod
    def backward(self, grad_backward):
        '''Propagate output through the loss.'''
        pass
    
class MseLoss(LossFunction):
    # mean squared error loss
    
    def forward(self, y_model, y_true):
        loss = np.sum(0.5 * (y_model-y_true)**2)
        return loss

    def backward(self, y_model, y_true):
        return y_model - y_true

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        '''The cross-entropy loss is trained using logits as the model output,
           i.e. the softmax probabilities are not computed during training.'''
        self.use_logits = True
    
    def forward(self, logits, y_true):
        # input:  model outputs (logits) and true class indices
        # output: softmax cross-entropy loss
        assert(y_true.dtype == np.uint8)
        true_class_logits = logits[np.arange(len(logits)), y_true]
        
        cross_entropy = - true_class_logits + np.log(np.sum(np.exp(logits), axis=-1))
        return cross_entropy

    def backward(self, logits, y_true):
        # convert to one-hot-encoding:
        ones_true_class = np.zeros_like(logits)
        ones_true_class[np.arange(len(logits)),y_true] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        
        return -ones_true_class + softmax
