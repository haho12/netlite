import numba
import numpy as np

from multiprocessing import Pool

#@numba.guvectorize(['f4(f4[:,:,:,:], f4[:,:,:,:], f4[:],i64,i64,i64,i64,i64)'], "(m,n),(n,p)->(m,p)")
def conv_wsum(X, W, b, ni, i, j, c, co):
    k = 5
    wsum = 0 # np.float32(b[co])

    for wi in range(k):
        for wj in range(k):
            for ci in range(c):
                wsum += W[wi,wj,ci,co] * X[ni,i+wi,j+wj,ci]
    return wsum

#@numba.guvectorize([(numba.float32[:,:,:,:], numba.float32[:,:,:,:], numba.int64, numba.int64, numba.int64, numba.int64, numba.int64, numba.int64, numba.float32)], '(),(),(),(),(),(),(),()->()')
#def conv_w(W, X, wi, ci, co, ni, i, j, wsum):
#    for wj in range(k):
#        wsum += W[wi,wj,ci,co] * X[ni,i+wi,j+wj,ci]#
#
#    #for i in range(x.shape[0]):
#    #    res[i] = x[i] + y

@numba.njit('f4[:,:,:,:](f4[:,:,:,:], f4[:,:,:,:], f4[:])', parallel=True)
def conv(X, W, b):
    k = 5
    n, h, w, c = X.shape
    h_out = h - (k - 1)
    w_out = w - (k - 1)
    c_out = W.shape[-1]
    
    output = np.zeros((n, h_out, w_out, c_out), dtype=np.float32)
    weight = W

    for ni in numba.prange(n):
    #for ni in range(n):
        for i in range(h_out):
            for j in range(w_out):
                for co in range(c_out):
                    wsum = b[co]
                    #wsum = conv_wsum(X, W, b, ni, i, j, c, co)
                    for ci in range(c):
                        for wi in range(k):
                            for wj in range(k):
                                wsum += W[wi,wj,ci,co] * X[ni,i+wi,j+wj,ci]

                    output[ni, i, j, co] = wsum

    return output

@numba.njit(parallel=True) # 'f4[:,:,:,:](f4[:,:,:,:], f4[:,:,:,:], f4[:])') # , parallel=True)
def conv_refp(X, W, b):
    k = 5
    n, h, w, c = X.shape
    h_out = h - (k - 1)
    w_out = w - (k - 1)
    c_out = W.shape[-1]

    output = np.zeros((n, h_out, w_out, c_out), dtype=np.float32)
    weight = W.copy().reshape(-1, c_out)

    for i in numba.prange(h_out):
    #for i in range(h_out):
        for j in range(w_out):
            inp = X[:, i:i+k, j:j+k, :].copy().reshape(n, -1)
            out = inp.dot(weight) + b
            output[:, i, j, :] = out.reshape(n, -1)
    
    return output

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


X = np.random.randn(100, 32, 32, 10).astype(np.float32)
W = np.random.normal(0.0, 1.0, size=(5, 5, 10, 6)).astype(np.float32)
b = np.random.normal(0.0, 1.0, size=(6)).astype(np.float32)

k = 5
n, h, w, c = X.shape
h_out = h - (k - 1)
w_out = w - (k - 1)
c_out = W.shape[-1]


%timeit out = conv_refp(X, W, b)
%timeit out_ref = conv_ref(X, W, b)

out      = conv_refp(X, W, b)
out_ref  = conv_ref(X, W, b)
err = np.abs(out - out_ref).max()
print(f"Error: {err}")
