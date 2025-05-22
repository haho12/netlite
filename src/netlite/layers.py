import numpy as np
import numba
from abc import ABC, abstractmethod

class Layer(ABC):
    '''Interface for neural network layers'''

    @abstractmethod
    def forward(self, X):
        '''Propagate input forward through the layer.'''
        pass

    @abstractmethod
    def backward(self, grad_backward):
        '''Propagate gradient backward through the layer.'''
        pass

    def get_weights(self):
        '''Return a dictionary of named trainable parameters.
           The parameters are returned by reference to a numpy array
           to be updated by the optimizer.'''
        return {} # default: no trainable parameters

    def get_gradients(self):
        '''Return a dictionary of named gradients for each trainable parameter.'''
        return {} # default: no trainable parameters

class ReLU(Layer):
    '''Rectified linear unit activation'''
    def forward(self, X):
        self.X = X
        return np.maximum(X, 0)
    
    def backward(self, grad_backward):
        relu_gradient = self.X > 0 
        return grad_backward * relu_gradient

class LeakyReLU(Layer):
    '''Rectified linear unit activation with a leak-factor'''
    leak_factor = np.float32(0.1)
    
    def forward(self, X):
        self.X = X
        return ((X>0) + self.leak_factor*(X<0)) * X

    def backward(self, grad_backward):
        return ((self.X>0) + self.leak_factor*(self.X<0)) * grad_backward

class Sigmoid(Layer):
    '''Sigmoid activation'''
    def forward(self, X):
        self.X = X
        self.Y = 1/(1 + np.exp(-X))
        return self.Y
    
    def backward(self, grad_backward):
        df = self.Y * (1 - self.Y)
        return df * grad_backward

class Softmax(Layer):
    '''Softmax classifier'''
    def forward(self, X):
        softmax = np.exp(X) / np.exp(X).sum(axis=-1,keepdims=True)
        return softmax
    
    def backward(self, grad_backward):
        raise SystemExit("Error: Softmax backpropagation is not efficient and " + 
                         "numerically less stable. " +
                         "Use CrossEntropyLoss with logits instead!")

class Flatten(Layer):
    '''Flatten an input tensor to a vector'''
    def forward(self, X):
        self.X_shape = X.shape
        return X.copy().reshape(X.shape[0], -1)

    def backward(self, grad_backward):
        return grad_backward.reshape(self.X_shape)
    
class FullyConnectedLayer(Layer):
    '''Fully connected aka Dense layer'''
    def __init__(self, n_inputs, n_outputs):
        # n_inputs:  number of input channels
        # n_outputs: number of output channels

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.initialize_weights()

    def initialize_weights(self):
        stddev = np.sqrt(2.0 / (self.n_inputs)) # msra
        self.weights = np.random.normal(0.0, stddev, size=(self.n_inputs, self.n_outputs)).astype('f')
        self.bias = np.zeros((1, self.n_outputs), dtype='f')

    def forward(self, X):
        self.X = X
        return X @ self.weights + self.bias
    
    def backward(self, grad_backward):
        # store gradients of weights for update
        self.grad_weights = self.X.T @ grad_backward
        self.grad_bias = np.sum(grad_backward, axis=0, keepdims=True)

        # back-propagate gradients
        grad_input = grad_backward @ self.weights.T # chain-rule
        return grad_input
    
    def get_weights(self):
        # return a dictionary of named parameters
        return {'weights' : self.weights, 'bias': self.bias}

    def get_gradients(self):
        # return a dictionary of named gradients
        return {'weights' : self.grad_weights, 'bias': self.grad_bias}
            
class ConvolutionalLayer(Layer):
    def __init__(self, kernel_width, inc, outc):
        self.kernel_width = kernel_width
        self.inc = inc
        self.outc = outc
        self.initialize_weights()
        
    def initialize_weights(self):
        k = self.kernel_width
        stddev = np.sqrt(2.0 / (k**2 * self.inc)) # msra
        self.weights = np.random.normal(0.0, stddev, size=(k, k, self.inc, self.outc)).astype('f')
        self.bias = np.zeros(self.outc, dtype='f')

    @staticmethod
    @numba.njit(parallel=True)
    def conv(X, weight, output, h_out, w_out, n, k):
        for i in numba.prange(h_out):
            for j in range(w_out):
                inp = X[:, i:i+k, j:j+k, :].copy().reshape(n, -1)
                out = inp.dot(weight)
                output[:, i, j, :] = out.reshape(n, -1)
    
    def forward(self, X):
        self.X = X
        k = self.kernel_width
        n, h, w, c = X.shape
        h_out = h - (k - 1)
        w_out = w - (k - 1)

        output = np.tile(self.bias, (n, h_out, w_out, 1))
        weights = self.weights.reshape(-1, self.outc)
        self.conv(X, weights, output, h_out, w_out, n, k)
        
        return output

    def backward(self, grad_backward):
        n, h, w, c = grad_backward.shape
        k = self.kernel_width
        h_in = h + (k - 1)
        w_in = w + (k - 1)

        self.grad_weights = np.zeros((k, k, self.inc, self.outc), dtype='f')
        for i in range(k):
            for j in range(k):
                # inp = (n, h, w, cin) => (n*h*w, cin) => (cin, n*h*w)
                inp = self.X[:, i:i+h, j:j+w, :].reshape(-1, self.inc).T
                # diff = (n, h, w, cout) => (n*h*w, cout)
                diff_out = grad_backward.reshape(-1, self.outc)
                self.grad_weights[i, j, :, :] = inp.dot(diff_out)
        self.grad_bias = np.sum(grad_backward, axis=(0, 1, 2))

        pad = k - 1
        grad_backward_pad = np.pad(grad_backward, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        rotated_weight = self.weights[::-1, ::-1, :, :].transpose(0, 1, 3, 2).reshape(-1, self.inc)
        grad_input = np.zeros((n, h_in, w_in, self.inc), dtype='f')
        self.conv(grad_backward_pad, rotated_weight, grad_input, h_in, w_in, n, k)

        return grad_input
    
    def get_weights(self):
        # return a dictionary of named parameters
        return {'weights' : self.weights, 'bias': self.bias}

    def get_gradients(self):
        # return a dictionary of named gradients
        return {'weights' : self.grad_weights, 'bias': self.grad_bias}

class MaxPoolingLayer(Layer):
    def forward(self, X):
        n, h, w, c = X.shape
        assert(h%2==0 and w%2==0)
        X_grid = X.reshape(n, h // 2, 2, w // 2, 2, c)
        out = np.max(X_grid, axis=(2, 4))
        self.mask = (out.reshape(n, h // 2, 1, w // 2, 1, c) == X_grid)
        return out

    def backward(self, grad_backward):
        n, h, w, c = grad_backward.shape
        grad_backward_grid = grad_backward.reshape(n, h, 1, w, 1, c)
        return (grad_backward_grid * self.mask).reshape(n, h * 2, w * 2, c)

class AvgPoolingLayer(Layer):
    def forward(self, X):
        n, h, w, c = X.shape
        assert(h%2==0 and w%2==0)
        X_grid = X.reshape(n, h // 2, 2, w // 2, 2, c)
        out = np.mean(X_grid, axis=(2, 4))
        self.mask = np.ones_like(X_grid) * (1/4)
        return out

    def backward(self, grad_backward):
        n, h, w, c = grad_backward.shape
        grad_backward_grid = grad_backward.reshape(n, h, 1, w, 1, c)
        return (grad_backward_grid * self.mask).reshape(n, h * 2, w * 2, c)

class GlobalAvgPoolingLayer(Layer):
    def forward(self, X):
        n, h, w, c = X.shape
        out = np.mean(X, axis=(1, 2))
        self.mask = np.ones_like(X) * (1/(h*w))
        return out

    def backward(self, grad_backward):
        n, c = grad_backward.shape    
        return grad_backward.reshape(n, 1, 1, c) * self.mask

