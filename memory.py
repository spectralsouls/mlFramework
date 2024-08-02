import numpy as np


# Storage is where the data is held
class Buffer:
    def __init__(self, size:int, dtype:np.dtype):
        self.size = size
    
    def allocate(self):
        pass

    def copyin(self):
        pass

    def copyout(self):
        pass

    
