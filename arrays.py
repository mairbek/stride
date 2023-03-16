
class Array:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.shape = []
        x = payload
        while type(x) == list:
            self.shape.append(len(x))
            if len(x) == 0:
                break
            x = x[0]
        self.shape = tuple(self.shape)

    def __getitem__(self, idx):
        # TODO: check if idx is a slice
        return self.payload[idx]

    # TODO first implement broadcast_to
    # TODO next implement broadcast

def _zeros(shape):
    if len(shape) == 0:
        return 0
    if len(shape) == 1:
        return [0] * shape[0]
    return [_zeros(shape[1:]) for _ in range(shape[0])]

def zeros(shape):
    return Array(_zeros(shape))

def _broadcast_to(source, destination, source_shape, destination_shape):
    pass

def broacast_to(arr, shape):
    if arr.shape == shape:
        return arr
    if len(shape) < len(arr.shape):
        raise ValueError(
            "Cannot broadcast array of shape {} to shape {}".format(arr.shape, shape))
    padded_shape = arr.shape
    for i in range(len(shape) - len(arr.shape)):
        padded_shape = (1,) + padded_shape
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] != padded_shape[i] and padded_shape[i] != 1:
            raise ValueError(
                "Cannot broadcast array of shape {} to shape {}".format(arr.shape, shape))
    result = _zeros(shape)
    result = arr.payload
    for i in range(len(shape) - len(arr.shape)):
        result = [result]
    print("res", result)
    for axis, size in reversed(list(enumerate(shape))):
        if padded_shape[axis] == 1:
            axis_list = [result] * size
            result = [axis_list] if axis == 0 else [axis_list[i:i+size] for i in range(0, len(axis_list), size)]
    return Array(result)
