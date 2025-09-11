import numpy as np

class SequentialNetwork():
    def __init__(self, layers = []):
        self.layers = layers

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
            
    def print(self, input_shape=None):
        print(f'Feed-forward network with {len(self.layers)} layers:')
        
        if len(self.layers) == 0:
            print('- no layers')
            return
        
        if input_shape is None:
            if not hasattr(self.layers[0], 'X'):        
                print('forward() has not been called yet - provide input_shape to get layer size info')
                return
            input_shape = self.layers[0].X.shape
        
        Xin = np.zeros(input_shape, dtype='f')
        for layer in self.layers:
            Xin = layer.print(Xin)

class FeedForwardNetwork(SequentialNetwork):
    '''Implements a feed-forward network with parallel layers as
       a directed acyclig graph (DAG)'''
    def __init__(self, layers = []):
        self.layers = {}
        for layer in layers:
            self.add_layer(layers)
            
        self.forward_edges = {}
        self.backward_edges = {}

    def add_layer(self, layer, input1_name=[], input2_name=[]):
        assert layer.name not in self.layers.keys(), f'Layer name {layer.name} already exists.'
        assert input1_name in self.layers.keys(), f'Input layer 1 {input1_name} does not exist.'
        assert input2_name in self.layers.keys(), f'Input layer 2 {input2_name} does not exist.'
        self.layers[layer.name] = layer
        
        self.backward_edges[layer.name] = []
        self.backward_edges[layer.name].append(input1_name)

        self.forward_edges[input1_name].append(layer.name)

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
            
    def print(self, input_shape=None):
        print(f'Feed-forward network with {len(self.layers)} layers:')
        
        if len(self.layers) == 0:
            print('- no layers')
            return
        
        if input_shape is None:
            if not hasattr(self.layers[0], 'X'):        
                print('forward() has not been called yet - provide input_shape to get layer size info')
                return
            input_shape = self.layers[0].X.shape
        
        Xin = np.zeros(input_shape, dtype='f')
        for layer in self.layers:
            Xin = layer.print(Xin)
            