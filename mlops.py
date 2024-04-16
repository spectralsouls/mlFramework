import math
from tensor_buffer import Tensor
import numpy as np

class Sigmoid:
   def forward(x): return 1/ (1 + math.exp(-x))
   def backward(x): 
      sigma = 1 / (1 + math.exp(-x))
      return sigma * (1 - sigma)

def dotproduct(a,b): 
   if b.shape[0] > 1:
      out = []
      for i in range(b.shape[0]):
        assert a.shape == b[i].shape
        ret = sum(x * y for x,y in zip(a, b[i]))
        out.append(ret)
      return out
   else: return sum(x * y for x,y in zip(a,b))



a = dotproduct(np.array([0.1,0.5]),np.array(
      [[0.1, 0.3],
       [0.2,0.4]]))

print(a)
