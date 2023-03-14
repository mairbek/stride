
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