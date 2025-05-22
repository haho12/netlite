import numpy as np

class Optimizer():
    def __init__(self, name, loss_func, learning_rate):
        self.name = name
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        
        # ADAM parameters
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_t = 0
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
        if self.name == 'sgd':
            for layer in model.layers:
                layer_weights   = layer.get_weights()
                layer_gradients = layer.get_gradients()
                for key in layer_weights:
                    layer_weights[key] -= self.learning_rate * layer_gradients[key]

        elif self.name == 'adam':
            self.adam_t += 1
            for layer in model.layers:
                layer_weights   = layer.get_weights()
                layer_gradients = layer.get_gradients()
                
                for key in layer_weights:
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