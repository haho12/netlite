# Copyright (c): Hanno Homann, 2024-'25

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