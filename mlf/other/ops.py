from typing import Union
from enum import Enum, auto

class UnaryOps(Enum):
    EXP2 = auto(); LOG2 = auto(); CAST = auto(); BITCAST = auto(); SIN = auto()
    SQRT = auto(); NEG = auto(); RECIP = auto()
class BinaryOps(Enum):
    ADD = auto(); MUL = auto(); IDIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto()
    CMPNE = auto(); XOR = auto(); SHR = auto(); SHL = auto()
class TernaryOps(Enum):
    WHERE = auto(); MULACC = auto()
class ReduceOps(Enum):
    SUM = auto(); MAX = auto()
class LoadOps(Enum):
    EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto() 
    CUSTOM = auto(); ASSIGN = auto(); VIEW = auto() 
class BufferOps(Enum):
    LOAD = auto(); CONST = auto(); STORE = auto()
    
Op = Union[UnaryOps, BinaryOps, TernaryOps, ReduceOps, LoadOps, BufferOps]