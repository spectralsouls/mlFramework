import math

def sigmoid(x):
   out = 1 / (1 + math.exp(-x))
   return out

class Layer:
   def __init__(self, inp, num_nodes, bias):
      self.inp = inp
      self.num_inp = len(inp)
      self.num_nodes = num_nodes
      self.bias = bias       
    #  self.weights = get_weights(self.num_inp, self.num_nodes)

   def get_weights(self):
      weights = []
      num_weights = self.num_inp * self.num_nodes
      for i in range(num_weights):
         weights.append(0.1 * i + 0.1)
      return weights

   
layer1 = Layer([1,2], 2, 0.25)





