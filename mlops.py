import math


class Sigmoid:
   def forward(x): return 1/ (1 + math.exp(-x))
   def backward(x): 
      sigma = 1 / (1 + math.exp(-x))
      return sigma * (1 - sigma)

def sigmoid(x):
   out = 1 / (1 + math.exp(-x))
   return out

def dotproduct(x,y):
   out = 0
   for i in range(len(x)):
      out += x[i] * y[i]
   return out