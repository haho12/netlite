import numpy as np

class OptimizerSGD():
    ''''Default optimizer for stochastic gradient descent (SGD)'''

    def __init__(self, loss_func, learning_rate):
        self.loss_func = loss_func
        self.learning_rate = learning_rate

    def step(self, model, X, y_true, compute_accuracy=True, forward_only=False):
        use_logits = self.loss_func.use_logits
        
        # forward pass
        y_model = model.forward(X, use_logits)
        
        # compute accuracy
        metrics = {}
        if compute_accuracy:
            metrics['n_correct'] = self.calc_accuracy(y_model, y_true)
        
        # compute loss
        loss = self.loss_func.forward(y_model, y_true)
        
        if forward_only:
            # skip backward pass and update for validation data
            return loss.sum(), metrics

        # backward pass
        loss_gradient = self.loss_func.backward(y_model, y_true)

        model.backward(loss_gradient, use_logits)

        # update
        self.update(model)
            
        return loss.sum(), metrics
    
    def update(self, model):
        for layer in model.layers:
            layer_weights   = layer.get_weights()
            layer_gradients = layer.get_gradients()
            for key in layer_weights:
                layer_weights[key] -= self.learning_rate * layer_gradients[key]
                
    def calc_accuracy(self, y_model, y_true):
        if y_model.shape[1] == 1:
            # single output neuron: threshold output at 0.5
            # note: this assumes a Sigmoid activation function
            y_model_predicted = (y_model>0.5)
        else:
            # multiple outputs: get index of maximum
            y_model_predicted = y_model.argmax(axis=1)

        n_correct_predictions = np.sum(y_model_predicted == y_true)
        
        return n_correct_predictions

class OptimizerMomentum(OptimizerSGD):
    '''Momentum optimizer using the mean of gradients'''
    
    def __init__(self, loss_func, learning_rate, beta=0.9):
        super().__init__(loss_func, learning_rate)
        self.beta = beta   # decay rate for momentum
        self.m = None

    def update(self, model):
        if self.m == None:
            # initialize moments
            self.m = []
            for i, layer in enumerate(model.layers):
                layer_weights   = layer.get_weights()
                layer_gradients = layer.get_gradients()
                mi = {} #  moment  buffer for i-th layer
                for key in layer_weights:
                    mi[key] = np.zeros_like(layer_gradients[key])
                self.m.append(mi)

        for i, layer in enumerate(model.layers):
            layer_weights = layer.get_weights()
            layer_gradients = layer.get_gradients()

            for key in layer_weights:
                # update momentum (low-pass filtered gradients)
                self.m[i][key] = self.beta * self.m[i][key] + (1 - self.beta) * layer_gradients[key]

                # update using the smoothed gradients
                layer_weights[key] -= self.learning_rate * self.m[i][key]

class OptimizerADAM(OptimizerSGD):
    '''ADAM optimizer with adaptive moment estimation'''

    def __init__(self, loss_func, learning_rate):
        super().__init__(loss_func, learning_rate)

        self.beta1 = 0.9   # decay rate of first moment (mean of gradients)
        self.beta2 = 0.999 # decay rate of second moment (uncentered variance of gradients)
        self.t = 0         # time step (number of iteration)
        self.eps = 1e-8

    def update(self, model):
        if self.t == 0:
            # initialize moments
            self.m = []
            self.v = []
            for i, layer in enumerate(model.layers):
                layer_weights   = layer.get_weights()
                layer_gradients = layer.get_gradients()
                mi = {} #  moment  buffer for i-th layer
                vi = {} # variance buffer for i-th layer
                for key in layer_weights:
                    mi[key] = np.zeros_like(layer_gradients[key])
                    vi[key] = np.zeros_like(layer_gradients[key])
                self.m.append(mi)
                self.v.append(vi)
        
        self.t += 1
        for i, layer in enumerate(model.layers):
            layer_weights   = layer.get_weights()
            layer_gradients = layer.get_gradients()
            
            for key in layer_weights:
                self.m[i][key] = self.beta1  * self.m[i][key] + (1 - self.beta1) * layer_gradients[key] / (1 - self.beta1**self.t)
                self.v[i][key] = self.beta2  * self.v[i][key] + (1 - self.beta2) * np.power(layer_gradients[key], 2) / (1 - self.beta2**self.t)
    
                layer_weights[key] -= self.learning_rate / (np.sqrt(self.v[i][key]) + self.eps) * self.m[i][key]

def batch_handler(X, y, batchsize, shuffle=False):
    assert len(X) == len(y), f'Number of data {len(X)} does not match number of labels {len(y)}'
    batchsize = min(batchsize, len(y))

    if shuffle:
        idxs = np.random.permutation((len(y)))

    for start_idx in range(0, len(X) - batchsize + 1, batchsize):
        if shuffle:
            batch = idxs[start_idx:start_idx + batchsize]
        else:
            batch = slice(start_idx, start_idx + batchsize)

        yield X[batch,:], y[batch]
