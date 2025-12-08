# add parent folder with netlite source to path
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, parent_dir)

import netlite as nl

def test_save_and_load():
    model = nl.NeuralNetwork([
                nl.FullyConnectedLayer(n_inputs=1, n_outputs=1),
            ])

    const1 = 3.
    const2 = 4.
    model.layers[0].weights[0,0] = const1
    model.layers[0].bias[0]      = const2
    
    filename = 'pytest_params.npy'
    model.save(filename)
    
    model2 = nl.NeuralNetwork([
                nl.FullyConnectedLayer(n_inputs=1, n_outputs=1),
            ])
    model2.load(filename)
    os.remove(filename)

    assert model2.layers[0].weights[0,0] == const1
    assert model2.layers[0].bias[0]      == const2
    