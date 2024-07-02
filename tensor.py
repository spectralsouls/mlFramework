from __future__ import annotations
import numpy as np

# np.frombuffer vs np.array

class Function:
    def forward(self): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self): raise NotImplementedError(f"backward not implemented for {type(self)}")

    def apply(self, fxn, *x:tensor):
        pass


import functions as F

class tensor:
    def __init__(self, data, dtype=np.int32):
        self.data, self.dtype = data, dtype
        self.shape = np.array(data, dtype).shape
        self.size = () if len(self.shape) == 0 else len(data)

    @property
    def numpy(self): return np.array(self.data, self.dtype)

    def __getitem__(self, idx): 
        return np.array(self.data)[idx]
    def __repr__(self):
        return f"{self.data}"


    def exp(self): return np.exp(self.data)
    def log(self): return np.log(self.data)

    def relu(self): return F.Relu.forward(self)