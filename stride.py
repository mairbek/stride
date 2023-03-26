from arrays import Array, View

from math import prod


def zeros(shape):
    dim = prod(shape)
    flat = [0] * dim
    view = View(0, (1, ), (dim, ))
    return Array(flat, view).reshape(shape)
