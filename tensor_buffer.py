import numpy as np
from mlops import dotproduct

#  For now Tensors will be an interface for numpy arrays

# Make tensor subscriptable

# implement broadcasting

class Tensor:
    def __init__ (self, data:np.ndarray, requires_grad=None): 
        self.data = np.array(data)
        self.requires_grad = requires_grad

    @property
    def shape(self): return self.data.shape

    def __getitem__(self, idx): return self.data[idx]

    def __add__(self, x): return Tensor(self.data + x.data)
    def __sub__(self, x): return Tensor(self.data - x.data)
    def __mul__(self, x): return Tensor(self.data * x.data)


