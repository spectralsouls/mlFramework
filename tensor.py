from __future__ import annotations
import numpy as np
from typing import Union, List, Tuple
import functools

# **** all fxns involving shapes need to work with ints and tuples as paramaters ****

# np.frombuffer vs np.array
class Function:
    def __init__(self, *x:tensor):
         self.needs_grad = [t.requires_grad for t in x]
         self.requires_grad = True if any(self.needs_grad) else False
         if self.requires_grad: self.parents = x

    def forward(self, *args): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self, *args): raise NotImplementedError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn:Function, *x:tensor, **kwargs) -> tensor:
        ctx = fxn(*x,)
        ret = tensor(ctx.forward(*[t.data for t in x], **kwargs))
        ret.requires_grad = ctx.requires_grad
        ret.ctx,= ctx if ctx.requires_grad else None,
        return ret

import functions as F

def broadcasted(x) -> tensor:
        if not isinstance(x, tensor):
            x = tensor(x)
        return x

class tensor:
    def __init__(self, data:Union[List, Tuple], dtype=np.int32, requires_grad=False): # data takes constants too
        self.ctx, self.grad = None, None
        self.requires_grad = requires_grad
        if isinstance(data, np.ndarray): 
             self.data = [list(d for d in data)] if len(data.shape) > 0 else data.item() #for scalar arrays
        else: self.data = data
        self.data, self.dtype = data, dtype
        self.shape = np.array(data, dtype).shape
        #self.size = () if len(self.shape) == 0 else len(data)

    def numpy(self): return np.array(self.data)

     # unary ops
    def negative(self): return F.Negative.apply(self)
    def recip(self): return F.Reciprocal.apply(self)
    def sqrt(self): return F.Sqrt.apply(self)
    def exp(self): return F.Exp.apply(self)
    def log(self): return F.Log.apply(self)
    def sin(self): return F.Sin.apply(self)
    def relu(self): return F.Relu.apply(self)
    def sigmoid(self): return F.Sigmoid.apply(self)
     # binary ops
    def add(self, y): return F.Add.apply(self, broadcasted(y))
    def sub(self, y): return F.Sub.apply(self, broadcasted(y))
    def mul(self, y): return F.Mul.apply(self, broadcasted(y))
    def div(self, y): return F.Div.apply(self, broadcasted(y))

     # movement ops
    def reshape(self, newshape, *args): 
         shape = tuple((newshape,)) if isinstance(newshape, int) else tuple(newshape)
         if args is not None: shape += tuple(a for a in args)
         return F.Reshape.apply(self, shape=shape)
    
    def transpose(self): return F.Transpose.apply(self)
    def flip(self, axis=None): return F.Flip.apply(self, axis) #doesnt seem to work
    def pad(self, width, mode='constant', **kwargs): 
         return F.Pad.apply(self, width=width, mode=mode, **kwargs)
    def shrink(self, axis=None): return F.Shrink.apply(self, axis=axis)
    def expand(self, axis): return F.Expand.apply(self, axis=axis)

     # reduce ops
    def sum(self, axis=None): 
          return F.Sum.apply(self, axis=axis)


    @staticmethod
    def random(*shape): return tensor(np.random.random_sample(size=shape)) # TODO: make random work with tuples too

    # ml ops
    def linear(self, weights:tensor, bias=None):
         out = (self * weights.transpose()).sum(axis=1)
         return out + bias if bias is not None else out
    # backwards pass of linear is most likely wrong

    def mean(self, axis=None): # backwards pass seems to be incorrect native gives grad: 1, pytorch gives grad: 1 / denom
         num = self.sum()
         denom = functools.reduce(lambda x, y: x * y, self.shape) if len(self.shape) > 0 else 1
         return num.div(denom)
    
    def square(self): return self * self


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
                    yield from walk(i, visited)
                    visited.add(i)
              yield node  
         return list(walk(self, set()))
    
    def backwards(self):
        assert self.shape == (), f"tensor must be scalar"
        self.grad = tensor(1.0, requires_grad=False) #initial grad of 1
        for t in reversed(self.dfs()):
                if t.ctx is not None:
                    grads = t.ctx.backward(t.grad.data)
                    new_grads = [tensor(g) for g in grads] if len(t.ctx.parents) > 1 else [tensor(grads)]
                    for t, g in zip(t.ctx.parents, new_grads):
                         if t.requires_grad:
                              t.grad = g  if t.grad is None else (t.grad + g)

