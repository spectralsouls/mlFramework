import numpy as np

#  For now Tensors will be an interface for numpy arrays

class Tensor:
    def __init__ (self, data:np.ndarray, requires_grad=None): 
        self.data = np.array(data)
        self.requires_grad = requires_grad

    @property
    def shape(self): return self.data.shape