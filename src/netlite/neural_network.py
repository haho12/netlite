import numpy as np

class NeuralNetwork():
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
                print('forward() has not been called yet - provide test_input to get layer size info')
                return
            input_shape = self.layers[0].X.shape
        
        Xin = np.zeros(input_shape, dtype='f')
        for layer in self.layers:
            Xin = layer.print(Xin)
            