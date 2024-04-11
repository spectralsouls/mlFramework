import math

def sigmoid(x):
   out = 1 / (1 + math.exp(-x))
   return out

def dotproduct(x,y):
   out = 0
   for i in range(len(x)):
      out += x[i] * y[i]
   return out