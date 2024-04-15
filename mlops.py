import math
from tensor_buffer import Tensor

class Sigmoid:
   def forward(x): return 1/ (1 + math.exp(-x))
   def backward(x): 
      sigma = 1 / (1 + math.exp(-x))
      return sigma * (1 - sigma)

def dotproduct(a,b): return sum(x * y for x,y in zip(a,b)) 
