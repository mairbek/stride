from tensor import Storage
from tensor import Tensor

import arrays

def zeros(n: int, m: int = 1):
    s = Storage(n * m)
    return Tensor(s, (n, m), (m, 1))

def zero_array(shape):
    return arrays.zeros(shape)

def array(payload):
    return arrays.Array(payload)

def out(arr, ndim):
    return arrays.out(arr, ndim)

def broadcast_to(arr, shape):
    return arrays.broacast_to(arr, shape)