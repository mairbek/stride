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
        n = len(self.shape)
        assert len(slices) == n
        normalized_slices = list(slices)
        for i in range(n):
            if slices[i] is None:
                normalized_slices[i] = slice(0, 1, self.shape[i])
        result = View(self.padding, [0] * n, [0] * n)
        for i in range(n - 1, -1, -1):
            # TODO validate 1+ logic lol
            result.shape[i] = 1 + (normalized_slices[i][2] -
                                   normalized_slices[i][0] - 1) // normalized_slices[i][1]
            result.stride[i] = self.stride[i] * normalized_slices[i][1]
            result.padding += normalized_slices[i][0] * self.stride[i]
        return result

    def flat_idx(self, idx):
        result = 0
        for i in range(len(idx)):
            result += self.stride[i] * idx[i]
        result += self.padding
        return result


def _ndenumerate(idx, container, depth):
    if depth >= len(idx):
        return None
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in _ndenumerate(idx[:], i, depth+1):
                yield j
        else:
            yield tuple(idx), i
        idx[depth] += 1


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
            fi = self.view.flat_idx(idx)
            yield idx, self.flat[fi]

    def __iter__(self):
        for i in range(self.shape[0]):
            # don't unwrap the view
            yield self[i]

    def __eq__(self, arg):
        if isinstance(arg, (list, tuple)):
            other = _ndenumerate([0]*len(self.shape), arg, 0)
        elif isinstance(arg, Array):
            if self.shape != arg.shape:
                return False
            other = arg.ndenumerate()
        else:
            return False
        if other is None:
            return False
        for i, j in zip(self.ndenumerate(), other):
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

    def _run_to_list(self, acc, idx, depth):
        new_idx = idx[:]
        new_idx.append(0)
        if depth == len(self.shape) - 1:
            for i in range(self.shape[depth]):
                new_idx[-1] = i
                fi = self.view.flat_idx(new_idx)
                acc.append(self.flat[fi])
            return
        for i in range(self.shape[depth]):
            new_idx[-1] = i
            ptr = []
            acc.append(ptr)
            self._run_to_list(ptr, new_idx[:], depth+1)

    def to_list(self):
        acc = []
        self._run_to_list(acc, [], depth=0)
        return acc

    def reshape(self, *arg):
        shape = arg
        if isinstance(arg[0], (list, tuple)):
            shape = arg[0]
        return Array(self.flat, self.view.reshape(shape))

    def subrange(self, slices):
        return Array(self.flat, self.view.subrange(slices))

    def __repr__(self):
        return f"array({self.to_list()})"
