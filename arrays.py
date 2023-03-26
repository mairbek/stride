from functools import reduce
from operator import mul
import itertools


class View(object):
    def __init__(self, padding, stride, shape):
        self.padding = padding
        self.stride = stride
        self.shape = shape

    def __repr__(self):
        return f"View(padding={self.padding}, stride={self.stride}, shape={self.shape})"

    def reshape(self, shape):
        total_shape = reduce(mul, shape)
        total_view_shape = reduce(mul, self.shape)
        assert total_shape == total_view_shape
        strides = [0] * len(shape)
        strides[-1] = self.stride[-1]
        for i in range(len(shape)-1, 0, -1):
            strides[i-1] = strides[i] * shape[i]
        return View(self.padding, strides, shape)

    def subrange(self, slices):
        return Array(self.flat, self.view.subrange(slices))


class Array(object):
    """docstring for Array."""

    def __init__(self, flat, view):
        super(Array, self).__init__()
        self.flat = flat
        self.view = view

    @property
    def shape(self):
        return tuple(self.view.shape)

    def ndindex(self):
        return itertools.product(*[range(i) for i in self.shape])

    def ndenumerate(self):
        for idx in self.ndindex():
            flat_idx = 0
            for i in range(len(idx)):
                flat_idx += self.view.stride[i] * idx[i]
            flat_idx += self.view.padding
            yield idx, self.flat[flat_idx]

    def __iter__(self):
        for i in range(self.shape[0]):
            # don't unwrap the view
            yield self[i]

    def __eq__(self, other):
        # TODO only works for 1d arrays!
        for i, j in zip(self, other):
            if i != j:
                return False
        return True

    def __getitem__(self, idx) -> int:
        if isinstance(idx, int):
            idx = (idx,)
        if isinstance(idx, slice):
            idx = (idx, )
        assert len(idx) <= len(self.shape)
        nidx = self._normalize_idx(idx)
        result = self.subrange(nidx)
        # Unwrap the view by reducing dimensions.
        restride = []
        reshape = []
        for i in range(len(self.shape)):
            if result.view.shape[i] == 1 and isinstance(idx[i], int):
                continue
            restride.append(result.view.stride[i])
            reshape.append(result.view.shape[i])
        if len(reshape) == 0:
            return result.flat[result.view.padding]
        result.view = View(result.view.padding, restride, reshape)
        return result

    def _normalize_idx(self, idx):
        result = []
        for i in range(len(self.shape)):
            ii = idx[i] if i < len(idx) else None
            if ii is None:
                result.append((0, 1, self.shape[i]))
            elif isinstance(ii, int):
                result.append((ii, 1, ii+1))
            elif isinstance(ii, slice):
                start = ii.start if ii.start is not None else 0
                step = ii.step if ii.step is not None else 1
                stop = ii.stop if ii.stop is not None else self.shape[i]
                result.append((start, step, stop))
            else:
                raise ValueError("Invalid index")
        return result

    def reshape(self, *arg):
        shape = arg
        if isinstance(arg[0], (list, tuple)):
            shape = arg[0]
        return Array(self.flat, self.view.reshape(shape))

    def subrange(self, slices):
        n = len(self.shape)
        assert len(slices) == n
        normalized_slices = list(slices)
        for i in range(n):
            if slices[i] is None:
                normalized_slices[i] = slice(0, 1, self.shape[i])
        new_view = View(self.view.padding, [0] * n, [0] * n)
        for i in range(n - 1, -1, -1):
            # TODO validate 1+ logic lol
            new_view.shape[i] = 1 + (normalized_slices[i][2] -
                                     normalized_slices[i][0] - 1) // normalized_slices[i][1]
            new_view.stride[i] = self.view.stride[i] * normalized_slices[i][1]
            new_view.padding += normalized_slices[i][0] * self.view.stride[i]
        return Array(self.flat, new_view)


def lazy_range(start, stop, shape):
    flat = list(range(start, stop))
    padding = 0
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = strides[::-1]
    stride = View(padding, strides, shape)
    return Array(flat, stride)
