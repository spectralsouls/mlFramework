from __future__ import annotations
import numpy as np

# np.frombuffer vs np.array

class Function:
    def forward(self, *args): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self): raise NotImplementedError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn, *x:tensor):
        ret = fxn.forward(*[t.data for t in x])
        return ret

import functions as F

def broadcasted(x):
        if not isinstance(x, tensor):
            return tensor(x)

class tensor:
    def __init__(self, data, dtype=np.int32):
        self.data, self.dtype = data, dtype
        self.shape = np.array(data, dtype).shape
        self.size = () if len(self.shape) == 0 else len(data)

    @property
    def numpy(self): return np.array(self.data, self.dtype)

    def negative(self): return F.Negative.apply(self)
    def recip(self): return F.Reciprocal.apply(self)
    def sqrt(self): return F.Sqrt.apply(self)
    def exp(self): return F.Exp.apply(self)
    def log(self): return F.Log.apply(self)
    def sin(self): return F.Sin.apply(self)
    def relu(self): return F.Relu.apply(self)

    def add(self, x): return F.Add.apply(self, broadcasted(x)) # should add be able to take more than 2 tensors (e.g. a + b + c + ...)
    def sub(self, x): return F.Add.apply(self, broadcasted(-x))
    def mul(self, x): return F.Mul.apply(self, broadcasted(x))
    def div(self, x): return F.Mul.apply(self, broadcasted(x).recip())

    def __repr__(self):
        return f"{self.data}"
    def __getitem__(self, idx): 
        return np.array(self.data)[idx]
    def __add__(self, x): return self.add(x)
    def __sub__(self, x): return self.sub(x)
    def __mul__(self, x): return self.mul(x)
    def __truediv__(self, x): return self.div(x)



