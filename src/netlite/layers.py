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

    def print(self, Xin):
        '''Print layer properties.'''
        Xout = self.forward(Xin)
        print(f'- {self.__class__.__name__}:')
        print(f'  - in ={Xin.shape}')
        print(f'  - out={Xout.shape}')        
        return Xout

    def get_weights(self):
        '''Return a dictionary of named trainable parameters.
           The parameters are returned by reference to a numpy array
           to be updated by the optimizer.'''
        return {} # default: no trainable parameters

    def get_gradients(self):
        '''Return a dictionary of named gradients for each trainable parameter.'''
        return {} # default: no trainable parameters

class FullyConnectedLayer(Layer):
    '''Fully connected aka Dense layer
       - data order: (batch, channels)
    '''
    def __init__(self, n_inputs, n_outputs):
        # n_inputs:  number of input channels
        # n_outputs: number of output channels

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.initialize_weights()

    def initialize_weights(self):
        stddev = np.sqrt(2.0 / (self.n_inputs)) # msra initialization
        self.weights = np.random.normal(0.0, stddev, size=(self.n_inputs, self.n_outputs)).astype('f')
        self.bias = np.zeros((1, self.n_outputs), dtype='f')

    def forward(self, X):
        assert len(X.shape)==2, 'FullyConnectedLayer: input must have dim=2'
        assert X.shape[1] == self.n_inputs, 'FullyConnectedLayer: invalid number of input channels'
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
    '''Convolutional layer
       - data order: (batch, height, width, channels)
    '''
    def __init__(self, kernel_size, in_channels, out_channels):
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initialize_weights()
        
    def initialize_weights(self):
        k = self.kernel_size
        stddev = np.sqrt(2.0 / (k**2 * self.in_channels)) # msra initialization
        self.weights = np.random.normal(0.0, stddev, size=(k, k, self.in_channels, self.out_channels)).astype('f')
        self.bias = np.zeros(self.out_channels, dtype='f')

    @staticmethod
    @numba.njit(parallel=True)
    def conv(X, weight, output, h_out, w_out, n, k):
        for i in numba.prange(h_out):
            for j in range(w_out):
                inp = X[:, i:i+k, j:j+k, :].copy().reshape(n, -1)
                out = inp.dot(weight)
                output[:, i, j, :] = out.reshape(n, -1)
    
    def forward(self, X):
        assert len(X.shape)==4, 'ConvolutionalLayer: input must have dim=4'
        assert X.dtype==np.float32, 'ConvolutionalLayer: input dtype must be float32'
        self.X = X
        k = self.kernel_size
        n, h, w, c = X.shape
        h_out = h - (k - 1)
        w_out = w - (k - 1)

        output = np.tile(self.bias, (n, h_out, w_out, 1))
        weights = self.weights.reshape(-1, self.out_channels)
        self.conv(X, weights, output, h_out, w_out, n, k)
        
        return output

    def backward(self, grad_backward):
        n, h, w, c = grad_backward.shape
        k = self.kernel_size
        h_in = h + (k - 1)
        w_in = w + (k - 1)

        # compute gradients for weights
        self.grad_weights = np.zeros((k, k, self.in_channels, self.out_channels), dtype='f')
        for i in range(k):
            for j in range(k):
                # reshape input: (n, h, w, cin) --> (n*h*w, cin) --> (cin, n*h*w)
                Xr = self.X[:, i:i+h, j:j+w, :].reshape(-1, self.in_channels).T
                # reshape grad_backward: (n, h, w, cout) --> (n*h*w, cout)
                Gr = grad_backward.reshape(-1, self.out_channels)
                self.grad_weights[i, j, :, :] = Xr.dot(Gr)
        # compute gradients for bias
        self.grad_bias = np.sum(grad_backward, axis=(0, 1, 2))

        # compute and return gradients for input
        pad = k - 1
        grad_backward_pad = np.pad(grad_backward, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        rotated_weight = self.weights[::-1, ::-1, :, :].transpose(0, 1, 3, 2).reshape(-1, self.in_channels)
        grad_input = np.zeros((n, h_in, w_in, self.in_channels), dtype='f')
        self.conv(grad_backward_pad, rotated_weight, grad_input, h_in, w_in, n, k)
        return grad_input
    
    def get_weights(self):
        # return a dictionary of named parameters
        return {'weights' : self.weights, 'bias': self.bias}

    def get_gradients(self):
        # return a dictionary of named gradients
        return {'weights' : self.grad_weights, 'bias': self.grad_bias}

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
        self.X = X
        softmax = np.exp(X) / np.exp(X).sum(axis=-1, keepdims=True)
        return softmax
    
    def backward(self, grad_backward):
        raise SystemExit("Error: Softmax backpropagation is not efficient and " + 
                         "numerically less stable. " +
                         "Use CrossEntropyLoss with logits instead!")

class Flatten(Layer):
    '''Flatten an input tensor to a vector'''
    def forward(self, X):
        self.X = X
        return X.copy().reshape(X.shape[0], -1)

    def backward(self, grad_backward):
        return grad_backward.reshape(self.X.shape)

class MaxPoolingLayer(Layer):
    def forward(self, X):
        self.X = X
        n, h, w, c = X.shape
        assert h%2==0 and w%2==0, 'input width and height must be even'
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
        self.X = X
        n, h, w, c = X.shape
        assert h%2==0 and w%2==0, 'input width and height must be even'
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
        self.X = X
        n, h, w, c = X.shape
        out = np.mean(X, axis=(1, 2))
        self.mask = np.ones_like(X) * (1/(h*w))
        return out

    def backward(self, grad_backward):
        n, c = grad_backward.shape    
        return grad_backward.reshape(n, 1, 1, c) * self.mask

class BatchNorm(Layer):
    def forward(self, X):
        training_mode = (X.shape[0] > 1)
        self.beta = 0.9
        self.eps  = 1e-7
        
        if training_mode:
            # compute batch statistics
            if len(X.shape)==2:
                batch_mean = np.mean(X, axis=0, keepdims=True)
                batch_var = np.var(X, axis=0, keepdims=True)
            else:
                assert len(X.shape)==4, 'Expecting image tensor.'
                batch_mean = np.mean(X, axis=(0,1,2), keepdims=True)
                batch_var = np.var(X, axis=(0,1,2), keepdims=True)

            # soft updates
            if not hasattr(self, 'running_mean'):
                # initialization
                self.running_mean = batch_mean
                self.running_var  = batch_var
            else:
                self.running_mean = self.beta*self.running_mean + (1-self.beta)*batch_mean
                self.running_var  = self.beta*self.running_var  + (1-self.beta)*batch_var

            # normalize
            x_out = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
        else:
            # Use running averages at inference
            assert hasattr(self, 'running_var'), 'BatchNorm must be called in training-mode first.'
            x_out = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return x_out

    def backward(self, grad_backward):
        # gradient wrt input
        assert hasattr(self, 'running_var'), 'BatchNorm forward must be called first.'
        dx = grad_backward / np.sqrt(self.running_var + self.eps)

        return dx
