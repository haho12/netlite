# Copyright (c): Hanno Homann, 2024-'25

import numpy as np
import time
import matplotlib.pyplot as plt

import netlite as nl

def batch_handler(X, y, batchsize, shuffle=False):
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
    
def train(model, optimizer, X_train, y_train, X_valid=(), y_valid=(), n_epochs=10, batchsize=32):
    log = {}
    log['loss_train'] = []
    log['loss_valid'] = []
    log['acc_train']  = []
    log['acc_valid']  = []
    for epoch in range(n_epochs):
        start_time = time.time()

        loss_sum = 0
        n_correct_predictions_sum = 0
        for x_batch, y_batch in batch_handler(X_train, y_train, batchsize=batchsize, shuffle=True):
            loss, n_correct_predictions = optimizer.step(model, x_batch, y_batch)
            loss_sum += loss
            n_correct_predictions_sum += n_correct_predictions

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"runtime: {elapsed_time:.1f} sec")
        
        loss_train_mean = loss_sum / len(y_train)
        log['loss_train'].append(loss_train_mean)
        log['acc_train'].append(n_correct_predictions_sum / len(y_train))

        if len(X_valid) > 0: # if validation data is available
            loss_sum_valid, n_correct_predictions_valid = optimizer.step(model, X_valid, y_valid, forward_only=True)
            loss_valid_mean = loss_sum_valid / len(y_valid)
            log['loss_valid'].append(loss_valid_mean)
            log['acc_valid'].append(n_correct_predictions_valid / len(y_valid))
            print(f'Epoch {epoch+1:3d} : loss_train {loss_train_mean:7.5f}, loss_valid {loss_valid_mean:7.5f}, acc_train {log["acc_train"][-1]:5.3f}, acc_valid {log["acc_valid"][-1]:5.3f}')
        else:
            print(f'Epoch {epoch+1:3d} : loss_train {loss_train_mean:7.1f}, acc_train {log["acc_train"][-1]:5.3f}')

        # re-initialize ADAM optimizer after each epoch to improve stability
        optimizer.adam_t = 0
    
    return log

if __name__ == '__main__':
    #testcase = 'xor'
    testcase = 'mnist_fcn'   # fast fully-connected network, more overfitting
    testcase = 'mnist_lenet' # original LeNet CNN
    
    if testcase == 'xor':
        model = nl.NeuralNetwork()
        use_sigmoid = True
        if use_sigmoid:
            np.random.seed(1)
            learning_rate = 2 # for Sigmoid activation function
            model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=2, n_outputs=2))
            model.add_layer(nl.layers.Sigmoid())
            model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=2, n_outputs=1))
            model.add_layer(nl.layers.Sigmoid())
        else:
            np.random.seed(2)
            learning_rate = 0.1 # for ReLU activation function
            model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=2, n_outputs=2))
            model.add_layer(nl.layers.LeakyReLU())
            model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=2, n_outputs=1))
            model.add_layer(nl.layers.LeakyReLU())
    
        loss_func = nl.MseLoss()
        
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
        X_train, y_train = nl.dataloader_mnist.load_train(num_images = 60000)
        X_test,  y_test  = nl.dataloader_mnist.load_valid(num_images = 10000)
        
        X_train = X_train[:,2:-2,2:-2,0].reshape(X_train.shape[0], -1)
        X_test  = X_test[:,2:-2,2:-2,0].reshape(X_test.shape[0], -1)

        # show some numbers
        #fig, ax = plt.subplots(1,12, figsize=(12,1), dpi=100)
        #for axis, idx in zip(fig.axes, np.arange(0, 0+12)):
        #    axis.imshow(X_train[idx, :].reshape(28,28), cmap='gray')
        #    axis.axis('off')
        #plt.show()
        
        model = nl.NeuralNetwork()
        model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=28**2, n_outputs=100))
        model.add_layer(nl.layers.ReLU())
        model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=100, n_outputs=200))
        model.add_layer(nl.layers.ReLU())
        model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=200, n_outputs=10))
        model.add_layer(nl.layers.Softmax())

        loss_func = nl.CrossEntropyLoss()
        
        learning_rate = 0.001
        n_epochs  =  25
        batchsize =  100
        optim = 'adam'

    elif testcase == 'mnist_lenet':
        X_train, y_train = nl.dataloader_mnist.load_train(num_images = 60000)
        X_test,  y_test  = nl.dataloader_mnist.load_valid(num_images = 10000)

        # show some numbers
        #fig, ax = plt.subplots(1,12, figsize=(12,1), dpi=100)
        #for axis, idx in zip(fig.axes, np.arange(0, 0+12)):
        #    axis.imshow(X_train[idx, :].reshape(28,28), cmap='gray')
        #    axis.axis('off')
        #plt.show()

        model = nl.NeuralNetwork()
        model.add_layer(nl.layers.ConvolutionalLayer(5, 1, 6))
        model.add_layer(nl.layers.ReLU())
        model.add_layer(nl.layers.AvgPoolingLayer())
        model.add_layer(nl.layers.ConvolutionalLayer(5, 6, 16))
        model.add_layer(nl.layers.ReLU())
        model.add_layer(nl.layers.AvgPoolingLayer())
        model.add_layer(nl.layers.Flatten())
        model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=400, n_outputs=120))
        model.add_layer(nl.layers.ReLU())
        model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=120, n_outputs=84))
        model.add_layer(nl.layers.ReLU())
        model.add_layer(nl.layers.FullyConnectedLayer(n_inputs=84, n_outputs=10))
        model.add_layer(nl.layers.Softmax())

        loss_func = nl.CrossEntropyLoss()
        
        learning_rate = 0.001
        n_epochs  =  20 # 20 is enough, test acc stagnates at ~98.0%
                        # 50 for AvgPooling, test acc <=98.5%
        batchsize =  32
        optim = 'adam'

    optimizer = nl.Optimizer(optim, loss_func, learning_rate)
    log = train(model, optimizer, X_train, y_train, X_test, y_test, n_epochs, batchsize)

    plt.plot(log['loss_train'], label='training loss')
    plt.plot(log['loss_valid'], label='validation loss')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    plt.plot(log['acc_train'], label='training accuracy')
    plt.plot(log['acc_valid'], label='validation accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
