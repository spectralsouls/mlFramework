import numpy as np

class Tensor:
    def __init__ (self, inp):
        self.inp = inp
    
    def create(self): return np.array(self.inp)


a = Tensor(1)
b = a.create()
c = a.create() + b
print(c)