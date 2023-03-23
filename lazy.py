# TODO better name
class StridedShape(object):
    def __init__(self, offset, stride, shape):
        self.offset = offset
        self.stride = stride
        self.shape = shape


class LazyArray(object):
    """docstring for LazyArray."""

    def __init__(self, flat, strided_shape):
        super(LazyArray, self).__init__()
        self.flat = flat
        self.strided_shape = strided_shape

    @property
    def shape(self):
        return self.strided_shape.shape

    def __getitem__(self, idx) -> int:
        print(idx)
        if isinstance(idx, int):
            idx = (idx,)
        assert len(idx) == len(self.shape)
        for i, s in zip(idx, self.shape):
            assert i < s
        # TODO oneliner
        flat_idx = list(idx)
        for i in range(len(idx)):
            flat_idx[i] = self.strided_shape.offset[i] + self.strided_shape.stride[i] * flat_idx[i]
        flat_idx = sum(flat_idx)
        return self.flat[flat_idx]

    def reshape(self, shape):
        # TODO: check if the shape is compatible
        # TODO move reshape calculation to StridedShape
        strides = [self.strided_shape.stride[-1]]
        for i in range(len(shape)-1, 0, -1):
            # TODO prealloc why append?
            strides.append(strides[-1]*shape[i])
        strides = strides[::-1]
        offset = self.strided_shape.offset
        return LazyArray(self.flat, StridedShape(offset, strides, shape))

def lazy_range(start, stop, shape):
    flat = list(range(start, stop))
    offset = [0] * len(shape)
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = strides[::-1]
    stride = StridedShape(offset, strides, shape)
    return LazyArray(flat, stride)