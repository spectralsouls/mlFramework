import numpy as np

# np.frombuffer vs np.array

class tensor:
    def __init__(self, data, dtype=np.int32, grad=None):
        self.data = data
        self.dtype = dtype
        self.shape = np.array(data, dtype).shape
        self.size = () if len(self.shape) == 0 else len(data)

    @property
    def numpy(self): return np.array(self.data, self.dtype)

    def __getitem__(self, idx): 
        return np.array(self.data)[idx]
