import numpy as np

class Numpy:
    def __init__(self, data, dtype):
        array = np.array(data, dtype=dtype) if not isinstance(data, np.ndarray) else data
        self.array, self.shape, self.dtype = array, array.shape, array.dtype

    def execute(self, op):
        pass

    def __repr__(self): return f"Numpy({self.array})"