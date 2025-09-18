import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# add parent folder with netlite source to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, parent_dir)
import netlite as nl

def training_loop(model, optimizer, X_train, y_train, n_epochs=2000):
    log = {}
    log['loss_train'] = []
    for epoch in range(n_epochs):
        loss_sum, _ = optimizer.step(model, X_train, y_train)
        
        loss_train_mean = loss_sum / len(y_train)
        log['loss_train'].append(loss_train_mean)

        print(f'Epoch {epoch+1:3d} : loss_train {loss_train_mean:7.4f}')
    
    return log

# create some random data points
x_train = np.arange(0, 20).reshape((-1,1))
#y_train = (x_train * m_gt + b_gt) + np.random.rand(num_points)*10
y_train = np.array([ -1.1773,   2.1500,  -0.1714,  8.5931,  5.9045,
                     11.0090,  10.5657,  18.9364, 23.4077, 18.3319,
                     29.3460,  28.9358,  34.6940, 34.6772, 39.4109,
                     39.2940,  46.8544,  46.7390,  46.6658,  53.2745]).reshape((-1,1))

y_train[3:5] -= 40 # outliers


# A simple linear model: y = w*x + b
model1 = nl.NeuralNetwork([
            nl.FullyConnectedLayer(n_inputs=1, n_outputs=1),
        ])
optimizer1 = nl.OptimizerSGD(loss_func=nl.MseLoss(), learning_rate=0.0001)
log1 = training_loop(model1, optimizer1, x_train, y_train)

model2 = nl.NeuralNetwork([
            nl.FullyConnectedLayer(n_inputs=1, n_outputs=1),
        ])
optimizer2 = nl.OptimizerSGD(loss_func=nl.HuberLoss(2.0), learning_rate=0.0001)
log2 = training_loop(model2, optimizer2, x_train, y_train)

model1_out = model1.forward(x_train)
model2_out = model2.forward(x_train)
plt.scatter(x_train, y_train, color='red', label='data')
plt.plot(x_train, model1_out, color='blue', label='MSE regression fit')
plt.plot(x_train, model2_out, color='green', label='Huber regression fit')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()

plt.plot(log1['loss_train'] / log1['loss_train'][0], label='MSE loss')
plt.plot(log2['loss_train'] / log2['loss_train'][0], label='Huber loss')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
