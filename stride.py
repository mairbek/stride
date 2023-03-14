from tensor import Storage
from tensor import Tensor
from arrays import Array

def zeros(n: int, m: int = 1):
    s = Storage(n * m)
    return Tensor(s, (n, m), (m, 1))

def array(payload):
    return Array(payload)