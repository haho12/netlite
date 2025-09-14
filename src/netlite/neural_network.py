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
                print('forward() has not been called yet - provide input_shape to get layer size info')
                return
            input_shape = self.layers[0].X.shape
        
        Xin = np.zeros(input_shape, dtype='f')
        for layer in self.layers:
            Xin = layer.print(Xin)
            
    def save(self, filename):
        model = {}
        for i, layer in enumerate(self.layers):
            weights = layer.get_weights()
            model[i] = weights
        np.save(filename, model)

    def load(self, filename):
        loaded_params = np.load(filename, allow_pickle=True).item()
        assert len(loaded_params) == len(self.layers), f'Error: Unexpected number of layers in file {filename}.'
        for i, layer in enumerate(self.layers):
            weights = layer.get_weights()
            for key in weights:
                assert key in loaded_params[i].keys(), f'Error: Key {key} no found in layer {i} of file {filename}.'
                weights[key] = loaded_params[i][key]
