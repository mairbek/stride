class Storage:
    def __init__(self, n: int) -> None:
        self.arr = [0] * n


accessor = slice | int


class Tensor:
    def __init__(self, storage: Storage, sizes: tuple[int, int], strides: tuple[int, int], offset: int = 0) -> None:
        self.storage = storage
        self.sizes = sizes
        self.strides = strides
        self.offset = offset

    def normalize_index(self, idx: accessor | tuple[accessor] | tuple[accessor, accessor]) -> tuple[slice, slice]:
        if isinstance(idx, int):
            idx = [idx,]
        if isinstance(idx, tuple):
            idx = list(idx)
        if len(idx) == 1:
            idx.append(slice(0, self.sizes[1]))
        if len(idx) != 2:
            raise ValueError("Invalid index")
        for i in range(2):
            if isinstance(idx[i], int):
                idx[i] = slice(idx[i], idx[i] + 1)
        return tuple(idx)

    def __getitem__(self, idx: accessor | tuple[accessor] | tuple[accessor, accessor]) -> int:
        norm_idx = self.normalize_index(idx)
#        if isinstance(idx, int):
#            # TODO check dims
#            if self.sizes[1] != 1:
#                raise ValueError("Not a 1D tensor")
#            return self.storage.arr[idx]
#        if isinstance(idx, tuple) and len(idx) == 2:
#            index = idx[0] * self.strides[0] + idx[1] * self.strides[1]
#            return self.storage.arr[index]
#        if isinstance(idx, slice):
#            # TODO check dims
#            if self.sizes[1] != 1:
#                raise ValueError("Not a 1D tensor")
#            start = idx.start if idx.start is not None else 0
#            stop = idx.stop if idx.stop is not None else self.sizes[0]
#            step = idx.step if idx.step is not None else 1
#            return [self.storage.arr[i] for i in range(start, stop, step)]

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
