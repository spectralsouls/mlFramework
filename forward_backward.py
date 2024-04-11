import math

def sigmoid(x):
   out = 1 / (1 + math.exp(-x))
   return out

def dotproduct(x,y):
   out = 0
   for i in range(len(x)):
      out += x[i] * y[i]
   return out

class Layer:
   def __init__(self, inp, num_nodes, bias):
      self.inp = inp
      self.num_inp = len(inp)
      self.num_nodes = num_nodes
      self.bias = bias       

   def get_weights(self):
      weights = []
      num_weights = self.num_inp * self.num_nodes
      for i in range(num_weights):
         weights.append(0.1 * i + 0.1)
      return weights

class Model:
   def __init__(self):
      self.layer1 = Layer([0.1, 0.5], 2, 0.25)
      self.weights1 = self.layer1.get_weights()
      self.layer2 = Layer([1,2], 2, 0.35)
   
   def forward(self):
      x = self.layer1.inp
      y = [0.1, 0.3]
      h1 = dotproduct(x, y) + self.layer1.bias
      print(h1)



nn = Model()

nn.forward()