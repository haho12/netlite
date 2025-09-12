import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# add parent folder with netlite source to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, parent_dir)
import netlite as nl

def training_loop(model, optimizer, X_train, y_train, n_epochs=10, batchsize=32):
    log = {}
    log['loss_train'] = []
    log['acc_train'] = []
    for epoch in range(n_epochs):
        loss_sum = 0
        n_correct_sum = 0
        for x_batch, y_batch in nl.batch_handler(X_train, y_train, batchsize=batchsize, shuffle=True):
            loss, metrics = optimizer.step(model, x_batch, y_batch)
            loss_sum += loss
            n_correct_sum += metrics['n_correct']
        
        loss_train_mean = loss_sum / len(y_train)
        log['loss_train'].append(loss_train_mean)
        log['acc_train'].append(n_correct_sum / len(y_train))

        print(f'Epoch {epoch+1:3d} : loss_train {loss_train_mean:7.4f}, acc_train {log["acc_train"][-1]:5.3f}')
    
    return log

### Set-up the model: ###
model = nl.NeuralNetwork([
            nl.FullyConnectedLayer(n_inputs=2, n_outputs=2),
            nl.Sigmoid(),
            nl.FullyConnectedLayer(n_inputs=2, n_outputs=1),
            nl.Sigmoid(),
        ])
model.print(input_shape=(4,2))

# input               x1 x2
X_train = np.array((( 0, 0),
                    ( 1, 0),
                    ( 0, 1),
                    ( 1, 1)))

# desired output: logical XOR
y_train = np.array((1,
                   0,
                   0,
                   1)).reshape((4,1))

optimizer = nl.OptimizerSGD(loss_func=nl.MseLoss(), learning_rate=2)
        
log = training_loop(model, optimizer, X_train, y_train, n_epochs=500, batchsize=4)

plt.plot(log['loss_train'], label='training')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

plt.plot(log['acc_train'], label='training')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid()
plt.show()
