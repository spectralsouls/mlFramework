from __future__ import annotations
import numpy as np
from typing import Union, List, Tuple

# np.frombuffer vs np.array
class Function:
    def __init__(self, *x:tensor):
         self.parents = x

    def forward(self, *args): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self, *args): raise NotImplementedError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn:Function, *x:tensor) -> tensor:
        ctx = fxn(*x)
        ret = tensor(ctx.forward(*[t.data for t in x]))
        ret.ctx = ctx
        return ret

import functions as F

def broadcasted(x) -> tensor:
        if not isinstance(x, tensor):
            x = tensor(x)
        return x

class tensor:
    def __init__(self, data:Union[List, Tuple], dtype=np.int32): # data takes constants too
        self.ctx = None
        self.grad = None
        self.data, self.dtype = data, dtype
        self.shape = np.array(data, dtype).shape
        self.size = () if len(self.shape) == 0 else len(data)

    
    def numpy(self): return np.array(self.data)

    def negative(self): return F.Negative.apply(self)
    def recip(self): return F.Reciprocal.apply(self)
    def sqrt(self): return F.Sqrt.apply(self)
    def exp(self): return F.Exp.apply(self)
    def log(self): return F.Log.apply(self)
    def sin(self): return F.Sin.apply(self)
    def relu(self): return F.Relu.apply(self)
    def sigmoid(self): return F.Sigmoid.apply(self)

    def add(self, x): return F.Add.apply(self, broadcasted(x))
    def sub(self, x): return F.Sub.apply(self, broadcasted(x))
    def mul(self, x): return F.Mul.apply(self, broadcasted(x))
    def div(self, x): return F.Mul.apply(self, broadcasted(x).recip()) # should there be a Div function?

    def __repr__(self):
        return f"tensor({self.data})"
    def __getitem__(self, idx): 
        return np.array(self.data)[idx]
    def __neg__(self): return self.negative()
    def __add__(self, x): return self.add(x)
    def __sub__(self, x): return self.sub(x)
    def __mul__(self, x): return self.mul(x)
    def __truediv__(self, x): return self.div(x)

    # depth first search
    def dfs(self):
         def walk(node, visited):
              visited.add(node)
              if node.ctx is not None:
                for i in node.ctx.parents:
                    #if i not in visited: 
                    yield from walk(i, visited)
                    visited.add(i)
              yield node  
         return list(walk(self, set()))
    

    def backwards(self):
        graph = reversed(self.dfs())
        self.grad = tensor(1.0) #initial grad of 1
        for t in graph:
                if t.ctx is not None:
                    new_grads = [tensor(t.ctx.backward(self.grad.data))]
                    print(new_grads)
                    for i,j in zip(t.ctx.parents, new_grads):
                        print(f"i: {i}, j: {j}")
                        i.grad = j
