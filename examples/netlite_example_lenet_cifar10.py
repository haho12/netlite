import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

# add parent folder with netlite source to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, parent_dir)
import netlite as nl
    
def training_loop(model, optimizer, X_train, y_train, X_valid=(), y_valid=(), n_epochs=10, batchsize=32):
    log = {}
    log['loss_train'] = []
    log['loss_valid'] = []
    log['acc_train']  = []
    log['acc_valid']  = []
    for epoch in range(n_epochs):
        
        ### Training ###
        start_time = time.time()
        loss_sum = 0
        n_correct_sum = 0
        for x_batch, y_batch in nl.batch_handler(X_train, y_train, batchsize=batchsize, shuffle=True):
            loss, metrics = optimizer.step(model, x_batch, y_batch)
            loss_sum += loss
            n_correct_sum += metrics['n_correct']

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"runtime: {elapsed_time:.1f} sec")
        
        loss_train_mean = loss_sum / len(y_train)
        log['loss_train'].append(loss_train_mean)
        log['acc_train'].append(n_correct_sum / len(y_train))

        ### Validation ###
        loss_sum = 0
        n_correct_sum = 0
        for x_batch, y_batch in nl.batch_handler(X_valid, y_valid, batchsize=batchsize, shuffle=False):
            loss, metrics = optimizer.step(model, x_batch, y_batch, forward_only=True)
            loss_sum += loss
            n_correct_sum += metrics['n_correct']

        loss_valid_mean = loss_sum / len(y_valid)
        log['loss_valid'].append(loss_valid_mean)
        log['acc_valid'].append(n_correct_sum / len(y_valid))
        print(f'Epoch {epoch+1:3d} : loss_train {loss_train_mean:7.4f}, loss_valid {loss_valid_mean:7.4f}, acc_train {log["acc_train"][-1]:5.3f}, acc_valid {log["acc_valid"][-1]:5.3f}')
    
    return log


X_train, y_train = nl.dataloader_cifar10.load_train(num_images = 50000)
X_test,  y_test  = nl.dataloader_cifar10.load_valid(num_images = 10000)

# show some numbers
fig, ax = plt.subplots(1, 6, figsize=(6,1), dpi=100)
for axis, idx in zip(fig.axes, np.arange(0, 6)):
    axis.imshow(X_train[idx, :, :, :], cmap='gray')
    axis.axis('off')
plt.show()

model = nl.NeuralNetwork([
            nl.BatchNorm(),
            nl.ConvolutionalLayer(5, 3, 32),
            nl.ReLU(),
            nl.MaxPoolingLayer(),
            nl.BatchNorm(),
            nl.ConvolutionalLayer(5, 32, 64),
            nl.ReLU(),
            nl.MaxPoolingLayer(),
            nl.Flatten(),
            nl.BatchNorm(),
            nl.FullyConnectedLayer(n_inputs=5*5*64, n_outputs=120),
            nl.ReLU(),
            nl.FullyConnectedLayer(n_inputs=120, n_outputs=84),
            nl.ReLU(),
            nl.FullyConnectedLayer(n_inputs=84, n_outputs=10),
            nl.Softmax(),
        ])
loss_func = nl.CrossEntropyLoss()

learning_rate = 0.001
n_epochs  =  20 # test acc at ~70% with MaxPooling
batchsize =  32

optimizer = nl.OptimizerADAM(loss_func, learning_rate)

log = training_loop(model, optimizer, X_train, y_train, X_test, y_test, n_epochs, batchsize)

plt.plot(log['loss_train'], label='training')
plt.plot(log['loss_valid'], label='validation')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training LeNet on CIFAR10 data')
plt.grid()
plt.show()

plt.plot(log['acc_train'], label='training')
plt.plot(log['acc_valid'], label='validation')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Training LeNet on CIFAR10 data')
plt.grid()
#plt.savefig("lenet_cifar10_acc_curves.svg")
plt.show()

