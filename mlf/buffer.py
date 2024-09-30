from __future__ import annotations
import numpy as np
from mlf.other.ops import UnaryOps, BinaryOps

class Numpy:
    def __init__(self, data, dtype):
        array = np.array(data, dtype=dtype) if not isinstance(data, np.ndarray) else data
        self.data, self.shape, self.dtype = array, array.shape, array.dtype

    def execute(self, op, *inp:Numpy):
        if isinstance(op, UnaryOps):
            x = self.data
            match op:
                case UnaryOps.EXP2:
                    ret = np.exp(x)
                case UnaryOps.LOG2:
                    ret = np.log(x)
                case UnaryOps.CAST:
                    pass
                case UnaryOps.BITCAST:
                    pass
                case UnaryOps.SIN:
                    ret = np.sin(x)
                case UnaryOps.SQRT:
                    ret = np.sqrt(x)
                case UnaryOps.NEG:
                    ret = np.negative(x)
                case UnaryOps.RECIP:
                    ret = np.reciprocal(x)
        elif isinstance(op, BinaryOps):
            x,y = self.data, inp[0].data
            match op:
                case BinaryOps.ADD:
                    ret = np.add(x, y)
                case BinaryOps.MUL:
                    ret = np.multiply(x, y)
                case BinaryOps.IDIV:
                    ret = np.divide(x, y)
                case BinaryOps.MAX:
                    pass
                case BinaryOps.MOD:
                    pass
                case BinaryOps.CMPLT:
                    pass
                case BinaryOps.CMPNE:
                    pass
                case BinaryOps.XOR:
                    pass
                case BinaryOps.SHR:
                    pass
                case BinaryOps.SHL:
                    pass
        return Numpy(ret, ret.dtype)

    def __repr__(self): return f"Numpy({self.data})"