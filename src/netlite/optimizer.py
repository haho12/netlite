import numpy as np

class OptimizerSGD():
    ''''Default optimizer for stochastic gradient descent (SGD)'''
    def __init__(self, loss_func, learning_rate):
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        
        # ADAM parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0
        self.eps = 1e-8

    def step(self, model, X, y_true, forward_only=False):
        use_logits = self.loss_func.use_logits
        
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
        
        if forward_only:
            # Skip backward pass and update for validation data
            return loss.sum(), n_correct_predictions

        # backward pass
        loss_gradient = self.loss_func.backward(y_model, y_true)

        model.backward(loss_gradient, use_logits)

        # update
        self.update(model)
            
        return loss.sum(), n_correct_predictions
    
    def update(self, model):
        for layer in model.layers:
            layer_weights   = layer.get_weights()
            layer_gradients = layer.get_gradients()
            for key in layer_weights:
                layer_weights[key] -= self.learning_rate * layer_gradients[key]

class OptimizerMomentum(OptimizerSGD):
    '''Momentum optimizer using the mean of gradients'''
    
    def __init__(self, loss_func, learning_rate, beta=0.9):
        super().__init__(loss_func, learning_rate)
        self.beta = beta   # decay rate for momentum

    def update(self, model):
        for layer in model.layers:
            layer_weights = layer.get_weights()
            layer_gradients = layer.get_gradients()

            for key in layer_weights:
                if key not in layer.m:
                    # initialize momentum buffer
                    layer.m[key] = np.zeros_like(layer_gradients[key])

                # update momentum (low-pass filtered gradients)
                layer.m[key] = self.beta * layer.m[key] + (1 - self.beta) * layer_gradients[key]

                # update using the smoothed gradients
                layer_weights[key] -= self.learning_rate * layer.m[key]

class OptimizerADAM(OptimizerSGD):
    '''ADAM optimizer with adaptive moment estimation'''
    def __init__(self, loss_func, learning_rate):
        super().__init__(loss_func, learning_rate)

        self.beta1 = 0.9   # decay rate of first moment (mean of gradients)
        self.beta2 = 0.999 # decay rate of second moment (uncentered variance of gradients)
        self.t = 0         # time step (number of iteration)
        self.eps = 1e-8

    def update(self, model):
        self.t += 1
        for layer in model.layers:
            layer_weights   = layer.get_weights()
            layer_gradients = layer.get_gradients()
            
            for key in layer_weights:
                if key not in layer.m:
                    # init moment and variance buffers
                    layer.m[key] = np.zeros_like(layer_gradients[key])
                    layer.v[key] = np.zeros_like(layer_gradients[key])
    
                layer.m[key] = self.beta1  * layer.m[key] + (1 - self.beta1) * layer_gradients[key] / (1 - self.beta1**self.t)
                layer.v[key] = self.beta2  * layer.v[key] + (1 - self.beta2) * np.power(layer_gradients[key], 2) / (1 - self.beta2**self.t)
    
                layer_weights[key] -= self.learning_rate / (np.sqrt(layer.v[key]) + self.eps) * layer.m[key]
