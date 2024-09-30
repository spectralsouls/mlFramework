from __future__ import annotations
import numpy as np
from typing import Union, List, Tuple
import functools, itertools
from mlf.buffer import Numpy

# **** all fxns involving shapes need to work with ints and tuples as paramaters ****
#              *** make -1 as a parameter work for the movement ops ***


# np.frombuffer vs np.array vs np.ndarray
class Function:
    def __init__(self, *x: tensor):
        self.needs_grad = [t.requires_grad for t in x]
        self.requires_grad = True if any(self.needs_grad) else False
        self.parents = x if self.requires_grad else None

    def forward(self, *args):
        raise NotImplementedError(f"forward not implemented for {type(self)}")

    def backward(self, *args):
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn: Function, *x: tensor, **kwargs) -> tensor:
        ctx = fxn(
            *x,
        )
        ret = tensor(ctx.forward(*[t.data for t in x], **kwargs))
        ret.requires_grad = ctx.requires_grad
        (ret.ctx,) = (ctx if ctx.requires_grad else None,)
        return ret


import mlf.functions as F


def broadcasted(x) -> tensor:
    if not isinstance(x, tensor):
        x = tensor(x)
    return x

#todo: self.data should be a memoryview
class tensor:
    def __init__(self, data:Union[List, np.ndarray, int, float], dtype=np.float32, requires_grad: bool = False):
        self.ctx, self.grad = None, None
        self.requires_grad = requires_grad
        # tensor class is the higher level api that the user uses, ndarrays will for now be the lower level abstraction
        data = data.data if isinstance(data, tensor) else data
        data = Numpy(data, dtype=dtype) if not isinstance(data, Numpy) else data
        self.data = data

    @property
    def shape(self):
        return self.data.shape
   
    @property
    def dtype(self):
        return self.data.dtype # replace this

    def numpy(self):
        return np.array(self.data.data) # fix this later

    # unary ops
    def negative(self):
        return F.Negative.apply(self)

    def recip(self):
        return F.Reciprocal.apply(self)

    def sqrt(self):
        return F.Sqrt.apply(self)

    def exp(self):
        return F.Exp.apply(self)

    def log(self):
        return F.Log.apply(self)

    def sin(self):
        return F.Sin.apply(self)

    def relu(self):
        return F.Relu.apply(self)

    def sigmoid(self):
        return F.Sigmoid.apply(self)

    # binary ops
    def negate(self, y):
        return not(F.Neq.apply(self, broadcasted(y)))

    def add(self, y):
        return F.Add.apply(self, broadcasted(y))

    def sub(self, y):
        return F.Sub.apply(self, broadcasted(y))

    def mul(self, y):
        return F.Mul.apply(self, broadcasted(y))

    def div(self, y):
        return F.Div.apply(self, broadcasted(y))

    # movement ops
    # all of these need to automatically turn parameter ints into tuples
    def reshape(self, newshape, *args):
        shape = tuple((newshape,)) if isinstance(newshape, int) else tuple(newshape)
        shape += tuple(a for a in args) if args is not None else None
        return F.Reshape.apply(self, shape=shape)

    def permute(self, axis):
        return F.Permute.apply(self, axis=axis)

    def flip(self, axis=0):
        return F.Flip.apply(self, axis=axis)

    def pad(self, width, mode="constant", **kwargs):
        return F.Pad.apply(self, width=width, mode=mode, **kwargs)

    def shrink(self, axis=None):
        return F.Shrink.apply(self, axis=axis)

    def expand(self, newshape: Tuple[int, ...]):
        if len(self.shape) < len(newshape):
            padded_shape = tuple(1 for _ in range(len(newshape) - len(self.shape))) + self.shape
            padded = self.reshape(padded_shape)
        else:
            padded = self
        return F.Expand.apply(padded, newshape=newshape)

    def transpose(self):
        order = tuple(sorted((i for i, _ in enumerate(self.shape)), reverse=True))
        return self.permute(order)

    # reduce ops
    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis=axis, keepdims=keepdims)

    # creation methods
    @staticmethod
    def random(shape: tuple, requires_grad=False):
        return tensor(np.random.uniform(size=shape), requires_grad=requires_grad)
    
    def zeroes(shape:tuple, requires_grad=False):
        return tensor(np.zeros(shape), requires_grad=requires_grad)
    
    def ones(shape:tuple, requires_grad=False):
        return tensor(np.ones(shape), requires_grad=requires_grad)

    # ml ops
    def linear(self, weights: tensor, bias=None):  # backwards pass of linear is most likely wrong
        out = (self * weights.transpose()).sum(axis=1)
        return out.add(bias) if bias is not None else out
    
    def batchnorm(self, mean, var, epsilon: float): #weights: tensor, bias: tensor, 
        num = self - mean.expand(self.shape)
        denom = (var + epsilon).sqrt()
        return num.div(denom)


    def mean(self, axis=None, keepdims=False):
        num = self.sum(axis=axis, keepdims=keepdims)
        if len(self.shape) > 0:
            vals = tuple(x for x,y in itertools.zip_longest(self.shape, num.shape, fillvalue=num.shape[0]) if x != y ) if len(num.shape) > 0 else self.shape
            denom = functools.reduce(lambda x,y: x*y, vals) 
        else:
            denom = 1
        return num.div(denom)

    def square(self):
        return self.mul(self)

    def __repr__(self):
        return f"tensor({self.data.data}, dtype={self.dtype})"  # TODO: make a better __repr__

    def __getitem__(self, idx):
        return tensor(self.data.data[idx])

    def __neg__(self):
        return self.negative()

    def __add__(self, x):
        return self.add(x)

    def __sub__(self, x):
        return self.sub(x)

    def __mul__(self, x):
        return self.mul(x)

    def __truediv__(self, x):
        return self.div(x)
    
    def __eq__(self, x):
        return self.negate(self != x)

    # depth first search
    def dfs(self):
        def walk(node, visited):
            visited.add(node)
            if node.ctx is not None:
                for i in node.ctx.parents:
                    if i not in visited:
                        yield from walk(i, visited)
                        visited.add(i)
            yield node

        return list(walk(self, set()))

    def backwards(self):
        assert self.shape == (), "tensor must be scalar"

        # implicit gradient creation
        self.grad = tensor(1.0, requires_grad=False)  # initial grad of 1 for the tensor that you called .backwards() on
        for t in reversed(self.dfs()):
            if t.ctx is not None:
                grads = t.ctx.backward(t.grad.data)
                if len(t.ctx.parents) > 1:
                    new_grads = [tensor(g) if g is not None else None for g in grads]
                else:
                    new_grads = [tensor(grads)]
                for t, g in zip(t.ctx.parents, new_grads):
                    if t.requires_grad:
                        t.grad = g if t.grad is None else (t.grad + g)
