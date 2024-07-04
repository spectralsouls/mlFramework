from __future__ import annotations
import numpy as np

# np.frombuffer vs np.array

class Function:
    def forward(self, *args): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self): raise NotImplementedError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn, x:tensor):
        ret = fxn.forward(x.data)
        return ret


import functions as F

class tensor:
    def __init__(self, data, dtype=np.int32):
        self.data, self.dtype = data, dtype
        self.shape = np.array(data, dtype).shape
        self.size = () if len(self.shape) == 0 else len(data)

    @property
    def numpy(self): return np.array(self.data, self.dtype)

    def exp(self): return F.Exp.apply(self)
    def log(self): return F.Log.apply(self)
    def relu(self): return F.Relu.apply(self)

    def __getitem__(self, idx): 
        return np.array(self.data)[idx]
    def __repr__(self):
        return f"{self.data}"


