import numpy as np
from typing import Tuple, Union


# A Buffer is a tensor without gradients
class Buffer:
    def __init__(self, base:Union[np.ndarray, list, tuple], shape:Tuple[int, ...], dtype): #base can take constants too
        self.base, self.shape, self.dtype = base, shape, dtype

    @staticmethod
    def create(base,):
        print(base)
        if not isinstance(base, np.ndarray):
            base = np.array(base)
        # from this point, base must be an ndarray
        return Buffer(base=base, shape=base.shape, dtype=base.dtype)

    def __repr__(self):
        return f"Buffer(shape:{self.shape}, dtype:{self.dtype})"
    

