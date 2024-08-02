import numpy as np


# Storage is where the data is held
class Buffer:
    def __init__(self, size:int, dtype:np.dtype, allocator):
        self.size = size
        self.allocator = allocator
    def allocate(self):
        self.allocator.aloc(size)

    def copyin(self):
        pass

    def copyout(self):
        pass

    
class Allocator:
    def alloc(self, size):
        assert size > 0, f"size must be positive, {size} < 0"