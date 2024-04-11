import math

def get_weights(num_weights):
         weights = []
         for i in range(num_weights):
            weights.append(0.1 * i + 0.1)
         return weights

def sigmoid(x):
   out = 1 / (1 + math.exp(-x))
   return out

class Model:
   def __init__(self, inputs, num_weights, bias):
      self.inputs = inputs
      self.bias = bias
      self.weights = get_weights(num_weights)



nn = Model(inputs = [0.1, 0.5], num_weights=8, bias=[0.25, 0.35])

