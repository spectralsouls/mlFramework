import numpy as np
import math

class Sigmoid:
   def forward(x): return 1/ (1 + np.exp(-x))
   def backward(x): 
      sigma = 1 / (1 + np.exp(-x))
      return sigma * (1 - sigma)

class MSE:
   def forward(observed, pred):
      assert len(observed) == len(pred)
      error = np.sum((observed - pred) ** 2 / len(observed))
      return error
   def backward(observed, pred):
      deriv = np.negative((observed - pred))
      return deriv