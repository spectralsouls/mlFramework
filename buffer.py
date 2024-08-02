import numpy as np
from typing import Tuple, Optional
import functools
from memory import Buffer


# A Buffer is a tensor without gradients
class LazyBuffer:
    def __init__(self, shape:Tuple[int, ...], dtype:np.dtype, base:Optional[Storage]=None):
        self.shape, self.dtype = shape, dtype
        self.size = functools.reduce(lambda x,y: x * y, self.shape)
        if base is None:
            self.buffer = Buffer(self.size, self.dtype)
        else:
            #assert base.base == base, f"the base must be its own base"
            self.base = base

    def __repr__(self): 
        return f"<LazyBuffer: shape {self.shape}, dtype{self.dtype}"
