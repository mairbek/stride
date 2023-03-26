from arrays import Array, View

from math import prod


def zeros(shape):
    dim = prod(shape)
    flat = [0] * dim
    view = View(0, (1, ), (dim, ))
    return Array(flat, view).reshape(shape)


def _flatten(shape, container, depth):
    if depth >= len(shape):
        shape.append(len(container))
    assert len(shape) == depth + 1
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in _flatten(shape, i, depth+1):
                yield j
        else:
            yield i


def array(container):
    shape = []
    flat = list(_flatten(shape, container, 0))
    view = View(0, (1, ), (prod(shape), ))
    return Array(flat, view).reshape(shape)


def arange(start, stop):
    return array(range(start, stop))
