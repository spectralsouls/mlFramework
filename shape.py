from typing import Tuple
from dataclasses import dataclass
import itertools, operator

#TODO: insert this error from pytorch (maybe): IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)

def get_strides(shape:Tuple[int, ...]) -> Tuple[int, ...]:
    strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))[::-1]
    return tuple(0 if s == 1 else st for s,st in zip(shape,strides))

@dataclass
class View:
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    offset: int
    contiguous: bool

    @staticmethod
    def create(shape:Tuple[int, ...], stride: Tuple[int, ...]=None, offset:int=0, contiguous:bool=True):
        assert isinstance(shape, tuple) and all(isinstance(s, int) for s in shape), f"the shape must be a tuple of ints"
        strides = get_strides(shape) if stride is None else stride
        return View(shape, strides, offset, contiguous)
    
    #can't use negative indexing, e.g. [-1], [-2] for any movement ops
    def permute(self, dims): 
        assert sorted(dims) == list(range(len(self.shape))), \
        f"dimensions don't match the desired ordering: {list(range(len(self.shape)))} != {sorted(dims)}"
        return View.create(tuple(self.shape[d] for d in dims), tuple(self.stride[d] for d in dims))
    
    def pad(self, dims): # and all((b>=0 and e >=0) for b,e in dims)    
        assert len(self.shape) == len(dims), f"input dimensions don't match pad dimensions: {self.shape} != {dims}"
        assert all(d >= 0 for d in dims), f"pad dimensions must be non-negative:{dims}"
        new_shape = tuple(s + d for s,d in zip(self.shape, dims))
        return View.create(new_shape, self.stride)

    def shrink(self, dims):
        pass

    def expand(self, dims):
        assert len(self.shape) == len(dims), \
        f"number of dimensions doesn't match length of desired dimensions: {len(self.shape)} != {len(dims)}"
        assert all(s == d or (s == 1 and st == 0) for s,d,st in zip(self.shape, dims, self.stride)), \
        f"expanded size {dims} must match the singleton dimension {self.shapes}"
        return View.create(shape=dims)
    
@dataclass
class ShapeTracker:
    views: Tuple[View, ...]
    