class Storage:
    def __init__(self, n: int) -> None:
        self.arr = [0] * n
class Tensor:
    # TODO also offset
    def __init__(self, storage: Storage, sizes: tuple[int, int], strides: tuple[int, int]) -> None:
        self.storage = storage
        self.sizes = sizes
        self.strides = strides

    def __getitem__(self, idx: int | tuple[int, int] | slice) -> int:
        if isinstance(idx, int):
            # TODO check dims
            if self.sizes[1] != 1:
                raise ValueError("Not a 1D tensor")
            return self.storage.arr[idx]
        if isinstance(idx, tuple) and len(idx) == 2:
            index = idx[0] * self.strides[0] + idx[1] * self.strides[1]
            return self.storage.arr[index]

        raise NotImplementedError

    def __setitem__(self, idx, val):
        print("set", idx, val)
        if isinstance(idx, int):
            if self.sizes[1] != 1:
                raise ValueError("Not a 1D tensor")
            self.storage.arr[idx] = val
            return
        if isinstance(idx, tuple) and len(idx) == 2:
            index = idx[0] * self.strides[0] + idx[1] * self.strides[1]
            self.storage.arr[index] = val
            return

        raise NotImplementedError