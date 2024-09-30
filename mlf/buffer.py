import numpy as np
from mlf.other.ops import UnaryOps, BinaryOps, ReduceOps

class Numpy:
    def __init__(self, data, dtype):
        array = np.array(data, dtype=dtype) if not isinstance(data, np.ndarray) else data
        self.data, self.shape, self.dtype = array, array.shape, array.dtype

    def execute(self, op):
        if isinstance(op, UnaryOps):
            match op:
                case UnaryOps.NEG:
                    return np.negative(self.data)
        elif isinstance(op, BinaryOps):
            pass
        elif isinstance(op, ReduceOps):
            pass


    def __repr__(self): return f"Numpy({self.data})"