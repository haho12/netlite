import numba
import numpy as np

# add parent folder with netlite source to path
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, parent_dir)
import netlite as nl

def conv_ref(X, W, b):
    k = 5
    n, h, w, c = X.shape
    h_out = h - (k - 1)
    w_out = w - (k - 1)
    c_out = W.shape[-1]

    output = np.zeros((n, h_out, w_out, c_out), dtype=np.float32)
    weight = W.reshape(-1, c_out)

    for i in range(h_out):
        for j in range(w_out):
            inp = X[:, i:i+k, j:j+k, :].reshape(n, -1)
            out = inp.dot(weight) + b
            output[:, i, j, :] = out.reshape(n, -1)
    
    return output

@numba.njit(parallel=True) # 'f4[:,:,:,:](f4[:,:,:,:], f4[:,:,:,:], f4[:])') # , parallel=True)
def conv_ref_parallel(X, W, b):
    k = 5
    n, h, w, c = X.shape
    h_out = h - (k - 1)
    w_out = w - (k - 1)
    c_out = W.shape[-1]

    output = np.zeros((n, h_out, w_out, c_out), dtype=np.float32)
    weight = W.copy().reshape(-1, c_out)

    for i in numba.prange(h_out):
        for j in range(w_out):
            inp = X[:, i:i+k, j:j+k, :].copy().reshape(n, -1)
            out = inp.dot(weight) + b # adding bias here in the ref implementation
            output[:, i, j, :] += out.reshape(n, -1)
    
    return output

def test_conv_netlite():    
    X = np.random.randn(100, 32, 32, 10).astype(np.float32)
    W = np.random.normal(0.0, 1.0, size=(5, 5, 10, 6)).astype(np.float32)
    b = np.random.normal(0.0, 1.0, size=(6)).astype(np.float32)
    k = 5
    n, h, w, c = X.shape
    c_out = W.shape[-1]

    layer = nl.ConvolutionalLayer(k, c, c_out)
    layer.weights = W
    layer.bias = b
    
    out = layer.forward(X)
    out_ref  = conv_ref(X, W, b)
    err = np.abs(out - out_ref).max()
    assert err < 0.00001

def test_conv_reference():
    X = np.random.randn(100, 32, 32, 10).astype(np.float32)
    W = np.random.normal(0.0, 1.0, size=(5, 5, 10, 6)).astype(np.float32)
    b = np.random.normal(0.0, 1.0, size=(6)).astype(np.float32)

    out      = conv_ref_parallel(X, W, b)
    out_ref  = conv_ref(X, W, b)
    err = np.abs(out - out_ref).max()
    assert err < 0.00001

#if __name__ == '__main__':
#test_conv_netlite()

# manual testing, ignored by pytest
#X = np.random.randn(100, 32, 32, 10).astype(np.float32)
#W = np.random.normal(0.0, 1.0, size=(5, 5, 10, 6)).astype(np.float32)
#b = np.random.normal(0.0, 1.0, size=(6)).astype(np.float32)

#k = 5
#n, h, w, c = X.shape
#h_out = h - (k - 1)
#w_out = w - (k - 1)
#c_out = W.shape[-1]    

# test timing... (unindent needed ?)
#%timeit out = conv_ref_parallel(X, W, b)
#%timeit out_ref = conv_ref(X, W, b)
    
#out      = conv_ref_parallel(X, W, b)
#out_ref  = conv_ref(X, W, b)
#err = np.abs(out - out_ref).max()
#print(f"Error: {err}")
