# Copyright (c): Hanno Homann, 2024-'25

import numpy as np
import numba
import time
import matplotlib.pyplot as plt
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
        
        # storage for adam momentums
        self.m = {}
        self.v = {}

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
        self.kernel = kernel_width
        k = kernel_width
        self.inc = inc
        self.outc = outc
        stddev = np.sqrt(2.0 / (k**2 * inc)) # msra
        self.weights = np.random.normal(0.0, stddev, size=(k, k, inc, outc)).astype('f')
        self.bias = np.zeros(outc, dtype='f')

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
        k = self.kernel
        n, h, w, c = X.shape
        h_out = h - (k - 1)
        w_out = w - (k - 1)

        output = np.tile(self.bias, (n, h_out, w_out, 1))
        weights = self.weights.reshape(-1, self.outc)
        self.conv(X, weights, output, h_out, w_out, n, k)
        
        return output

    def backward(self, grad_backward):
        n, h, w, c = grad_backward.shape
        k = self.kernel
        h_in = h + (k - 1)
        w_in = w + (k - 1)

        self.weights_diff = np.zeros((k, k, self.inc, self.outc), dtype='f')
        for i in range(k):
            for j in range(k):
                # inp = (n, h, w, cin) => (n*h*w, cin) => (cin, n*h*w)
                inp = self.X[:, i:i+h, j:j+w, :].reshape(-1, self.inc).T
                # diff = (n, h, w, cout) => (n*h*w, cout)
                diff_out = grad_backward.reshape(-1, self.outc)
                self.weights_diff[i, j, :, :] = inp.dot(diff_out)
        self.bias_diff = np.sum(grad_backward, axis=(0, 1, 2))

        pad = k - 1
        grad_backward_pad = np.pad(grad_backward, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        rotated_weight = self.weights[::-1, ::-1, :, :].transpose(0, 1, 3, 2).reshape(-1, self.inc)
        grad_input = np.zeros((n, h_in, w_in, self.inc), dtype='f')
        self.conv(grad_backward_pad, rotated_weight, grad_input, h_in, w_in, n, k)

        return grad_input

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
    
class MseLoss():
    # mean squared error loss
    
    def forward(self, y_model, y_true):
        loss = np.sum(0.5 * (y_model-y_true)**2)
        return loss

    def backward(self, y_model, y_true):
        return y_model - y_true

class CrossEntropyLoss():
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

class Optimizer():
    def __init__(self, loss_func, learning_rate):
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        
        # ADAM parameters
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_t = 0
        self.eps = 1e-8

    def step(self, model, X, y_true):
        use_logits = isinstance(self.loss_func, CrossEntropyLoss)
        
        # forward pass
        y_model = model.forward(X, use_logits)
        
        # compute accuracy
        if y_model.shape[1] > 1:
            # multiple outputs: get index of maximum
            y_model_maxidx = y_model.argmax(axis=1)
            n_correct_predictions = np.sum(y_model_maxidx == y_true)
        else:
            # single output: threshold at 0.5
            n_correct_predictions = np.sum((y_model>0.5) == y_true)
        
        # compute loss
        loss = self.loss_func.forward(y_model, y_true)
        
        # backward pass
        loss_gradient = self.loss_func.backward(y_model, y_true)

        model.backward(loss_gradient, use_logits)

        # update
        if self.optimizer == 'sgd':
            for layer in model.layers:
                layer_weights   = layer.get_weights()
                layer_gradients = layer.get_gradients()
                for key in layer_weights:
                    layer_weights[key] -= self.learning_rate * layer_gradients[key]

        elif self.optimizer == 'adam':
            self.adam_t += 1
            for layer in model.layers:
                layer_weights   = layer.get_weights()
                layer_gradients = layer.get_gradients()
                
                for key in layer_weights:
                    layer_weights[key] -= self.learning_rate * layer_gradients[key]
        
                    if self.adam_t == 1:
                        # init m and v
                        layer.m[key] = np.zeros_like(layer_gradients[key])
                        layer.v[key] = np.zeros_like(layer_gradients[key])
        
                    layer.m[key] = self.adam_beta1  * layer.m[key] + (1 - self.adam_beta1) * layer_gradients[key] / (1 - self.adam_beta1**self.adam_t)
                    layer.v[key] = self.adam_beta2  * layer.v[key] + (1 - self.adam_beta2) * np.power(layer_gradients[key], 2) / (1 - self.adam_beta2**self.adam_t)
        
                    layer_weights[key] -= self.learning_rate / (np.sqrt(layer.v[key]) + self.eps) * layer.m[key]
    
        else:
            raise SystemExit('Error: unknown optimizer ' + str(self.optimizer))
            
        return loss.sum(), n_correct_predictions

    def batch_handler(self, X, y, batchsize, shuffle=False):
        assert len(X) == len(y)
        batchsize = min(batchsize, len(y))

        if shuffle:
            idxs = np.random.permutation((len(y)))

        for start_idx in range(0, len(X) - batchsize + 1, batchsize):
            if shuffle:
                batch = idxs[start_idx:start_idx + batchsize]
            else:
                batch = slice(start_idx, start_idx + batchsize)

            yield X[batch,:], y[batch]
    
    def train(self, model, X_train, y_train, X_valid=(), y_valid=(),
              n_epochs=10, batchsize=32, optimizer='adam'):

        self.optimizer = optimizer

        self.log = {}
        self.log['loss_train'] = []
        self.log['acc_train']  = []
        self.log['acc_valid']  = []
        for epoch in range(n_epochs):
            start_time = time.time()

            loss_sum = 0
            n_correct_predictions_sum = 0
            for x_batch, y_batch in self.batch_handler(X_train, y_train, batchsize=batchsize, shuffle=True):
                loss, n_correct_predictions = self.step(model, x_batch, y_batch)
                loss_sum += loss
                n_correct_predictions_sum += n_correct_predictions

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"runtime: {elapsed_time:.1f} sec")
            self.log['loss_train'].append(loss_sum)
            self.log['acc_train'].append(n_correct_predictions_sum / len(y_train))

            if len(X_valid) > 0:
                y_model_valid  = model.forward(X_valid)
                
                if y_model_valid.shape[1] > 1:
                    # multiple outputs: get index of maximum
                    y_model_valid  = y_model_valid.argmax(axis=1)
                    n_correct_predictions_valid = np.mean(y_model_valid == y_valid)
                else:
                    # single output: threshold at 0.5
                    n_correct_predictions_valid = np.mean((y_model_valid>0.5) == y_valid)
                self.log['acc_valid'].append(n_correct_predictions_valid)
                print(f'Epoch {epoch+1:3d} : loss {loss_sum:7.1f}, acc_train {self.log["acc_train"][-1]:5.3f}, acc_valid {self.log["acc_valid"][-1]:5.3f}')
            else:
                print(f'Epoch {epoch+1:3d} : loss {loss_sum:7.1f}, acc_train {self.log["acc_train"][-1]:5.3f}')

            # re-initialize ADAM optimizer after each epoch to improve stability
            self.adam_t = 0

class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X, use_logits = False):
        if use_logits:
            # skip last activation layer
            end = len(self.layers) - 1
        else:
            end = len(self.layers)

        for layer in self.layers[:end]:
            X = layer.forward(X)
        return X

    def backward(self, gradient_backward, use_logits = False):
        if use_logits:
            # skip last activation layer
            start = len(self.layers) - 2
        else:
            start = len(self.layers) - 1

        for layer in self.layers[start::-1]:
            gradient_backward = layer.backward(gradient_backward)

if __name__ == '__main__':
    #testcase = 'xor'
    #testcase = 'mnist_fcn'   # fast fully-connected network, more overfitting
    testcase = 'mnist_lenet' # original LeNet CNN
    
    if testcase == 'xor':
        model = NeuralNetwork()
        use_sigmoid = True
        if use_sigmoid:
            np.random.seed(1)
            learning_rate = 2 # for Sigmoid activation function
            model.add_layer(FullyConnectedLayer(n_inputs=2, n_outputs=2))
            model.add_layer(Sigmoid())
            model.add_layer(FullyConnectedLayer(n_inputs=2, n_outputs=1))
            model.add_layer(Sigmoid())
        else:
            np.random.seed(2)
            learning_rate = 0.1 # for ReLU activation function
            model.add_layer(FullyConnectedLayer(n_inputs=2, n_outputs=2))
            model.add_layer(LeakyReLU())
            model.add_layer(FullyConnectedLayer(n_inputs=2, n_outputs=1))
            model.add_layer(LeakyReLU())
    
        loss_func = MseLoss()
        
        #                    x1 x2
        X_train = np.array((( 0, 0),
                            ( 1, 0),
                            ( 0, 1),
                            ( 1, 1)))
    
        # desired output: logical XOR
        y_train = np.array((1,
                           0,
                           0,
                           1)).reshape((4,1))
        
        X_test = X_train
        y_test = y_train
        
        batchsize = 4
        n_epochs = 1000
        optim = 'sgd'
        
    elif testcase == 'mnist_fcn':
        import dataloader_mnist

        X_train, y_train = mnist_loader.load_train(num_images = 60000)
        X_test,  y_test  = mnist_loader.load_valid(num_images = 10000)
        
        X_train = X_train[:,2:-2,2:-2,0].reshape(X_train.shape[0], -1)
        X_test  = X_test[:,2:-2,2:-2,0].reshape(X_test.shape[0], -1)

        # show some numbers
        #fig, ax = plt.subplots(1,12, figsize=(12,1), dpi=100)
        #for axis, idx in zip(fig.axes, np.arange(0, 0+12)):
        #    axis.imshow(X_train[idx, :].reshape(28,28), cmap='gray')
        #    axis.axis('off')
        #plt.show()
        
        model = NeuralNetwork()
        model.add_layer(FullyConnectedLayer(n_inputs=28**2, n_outputs=100))
        model.add_layer(ReLU())
        model.add_layer(FullyConnectedLayer(n_inputs=100, n_outputs=200))
        model.add_layer(ReLU())
        model.add_layer(FullyConnectedLayer(n_inputs=200, n_outputs=10))
        model.add_layer(Softmax())

        loss_func = CrossEntropyLoss()
        
        learning_rate = 0.001
        n_epochs  =  25
        batchsize =  100
        optim = 'adam'

    elif testcase == 'mnist_lenet':
        import dataloader_mnist

        X_train, y_train = mnist_loader.load_train(num_images = 60000)
        X_test,  y_test  = mnist_loader.load_valid(num_images = 10000)

        # show some numbers
        #fig, ax = plt.subplots(1,12, figsize=(12,1), dpi=100)
        #for axis, idx in zip(fig.axes, np.arange(0, 0+12)):
        #    axis.imshow(X_train[idx, :].reshape(28,28), cmap='gray')
        #    axis.axis('off')
        #plt.show()

        model = NeuralNetwork()
        model.add_layer(ConvolutionalLayer(5, 1, 6))
        model.add_layer(ReLU())
        model.add_layer(AvgPoolingLayer())
        model.add_layer(ConvolutionalLayer(5, 6, 16))
        model.add_layer(ReLU())
        model.add_layer(AvgPoolingLayer())
        model.add_layer(Flatten())
        model.add_layer(FullyConnectedLayer(n_inputs=400, n_outputs=120))
        model.add_layer(ReLU())
        model.add_layer(FullyConnectedLayer(n_inputs=120, n_outputs=84))
        model.add_layer(ReLU())
        model.add_layer(FullyConnectedLayer(n_inputs=84, n_outputs=10))
        model.add_layer(Softmax())

        loss_func = CrossEntropyLoss()
        
        learning_rate = 0.001
        n_epochs  =  50 # 20 is enough, test acc stagnates at ~98.0%
                        # 50 for AvgPooling, test acc <=98.5%
        batchsize =  32
        optim = 'adam'

    optimizer = Optimizer(loss_func, learning_rate)
    optimizer.train(model, X_train, y_train, X_test, y_test,
                  n_epochs, batchsize, optim)

    plt.plot(optimizer.log['loss_train'], label='training loss')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    plt.plot(optimizer.log['acc_train'], label='training accuracy')
    plt.plot(optimizer.log['acc_valid'],  label='valid accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
