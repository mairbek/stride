from functools import reduce
from operator import mul


class View(object):
    def __init__(self, offset, stride, shape):
        self.offset = offset
        self.stride = stride
        self.shape = shape


class LazyArray(object):
    """docstring for LazyArray."""

    def __init__(self, flat, view):
        super(LazyArray, self).__init__()
        self.flat = flat
        self.view = view

    @property
    def shape(self):
        return self.view.shape

    def __getitem__(self, idx) -> int:
        if isinstance(idx, int):
            idx = (idx,)
        assert len(idx) == len(self.shape)
        for i, s in zip(idx, self.shape):
            assert i < s
        # TODO oneliner
        flat_idx = list(idx)
        for i in range(len(idx)):
            flat_idx[i] = self.view.offset[i] + \
                self.view.stride[i] * flat_idx[i]
        flat_idx = sum(flat_idx)
        return self.flat[flat_idx]

    def reshape(self, shape):
        total_shape = reduce(mul, shape)
        total_view_shape = reduce(mul, self.view.shape)
        assert total_shape == total_view_shape
        strides = [0] * len(shape)
        strides[-1] = self.view.stride[-1]
        for i in range(len(shape)-1, 0, -1):
            strides[i-1] = strides[i] * shape[i]
        offset = self.view.offset
        return LazyArray(self.flat, View(offset, strides, shape))


def lazy_range(start, stop, shape):
    flat = list(range(start, stop))
    offset = [0] * len(shape)
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = strides[::-1]
    stride = View(offset, strides, shape)
    return LazyArray(flat, stride)
