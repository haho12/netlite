import numpy as np

# add parent folder with netlite source to path
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, parent_dir)
import netlite as nl

def test_batch_handler_keeps_last_batch():
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    batch_sizes = [len(yb) for _, yb in nl.batch_handler(X, y, batchsize=4)]
    assert batch_sizes == [4, 4, 2]

def test_batch_handler_shuffle_uses_every_sample_once():
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    seen = np.concatenate([yb for _, yb in nl.batch_handler(X, y, batchsize=4, shuffle=True)])
    assert sorted(seen) == list(range(10))

def test_batch_handler_drop_last():
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    batch_sizes = [len(yb) for _, yb in nl.batch_handler(X, y, batchsize=4, drop_last=True)]
    assert batch_sizes == [4, 4]

def test_batch_handler_exact_multiple():
    X = np.arange(8).reshape(8, 1)
    y = np.arange(8)
    for drop_last in (False, True):
        batch_sizes = [len(yb) for _, yb in nl.batch_handler(X, y, batchsize=4, drop_last=drop_last)]
        assert batch_sizes == [4, 4]
