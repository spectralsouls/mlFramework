import numpy as np

# Start with a Buffer, which holds the data and tracks the shape
# Then add a Tensor which inherits from the Buffer but also contains gradients 

class Tensor:
    def __init__ (self, data, requires_grad=None): 
        self.data = data
        self.requires_grad = requires_grad

    @property
    def create(self): return np.array(self.data)

   # def numpy(self) -> np.ndarray: return print(np.ndarray(shape=(3,1) , buffer=self.data))

