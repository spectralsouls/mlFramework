import math

class Function:
   def __init__(self, x):
      self.x = x

   def apply(fxn, x, state:int):
      if state == 1: fxn = forward
      else: fxn = backward
      return fxn(x)


class Sigmoid(Function):
   def forward(x): return 1/ (1 + math.exp(-x))
   def backward(x): 
      sigma = 1 / (1 + math.exp(-x))
      return sigma * (1 - sigma)

def dotproduct(x,y):
   out = 0
   for i in range(len(x)):
      out += x[i] * y[i]
   return out

a = Tensor(1)
print(a.create())

x = 1

a.add(x) == 2

F.Sigmoid.apply(a)