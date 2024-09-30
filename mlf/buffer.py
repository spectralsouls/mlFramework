import numpy as np
from mlf.other.ops import UnaryOps, BinaryOps, ReduceOps

class Numpy:
    def __init__(self, data, dtype):
        array = np.array(data, dtype=dtype) if not isinstance(data, np.ndarray) else data
        self.data, self.shape, self.dtype = array, array.shape, array.dtype

    def execute(self, op):
        if isinstance(op, UnaryOps):
            match op:
                case UnaryOps.EXP2:
                    return np.exp(self.data)
                case UnaryOps.LOG2:
                    return np.log(self.data)
                case UnaryOps.CAST:
                    pass
                case UnaryOps.BITCAST:
                    pass
                case UnaryOps.SIN:
                    return np.sin(self.data)
                case UnaryOps.SQRT:
                    return np.sqrt(self.data)
                case UnaryOps.NEG:
                    return np.negative(self.data)
                case UnaryOps.RECIP:
                    return np.reciprocal(self.data)
        elif isinstance(op, BinaryOps):
            pass
        elif isinstance(op, ReduceOps):
            pass


    def __repr__(self): return f"Numpy({self.data})"