import numpy as np
from mlops import Sigmoid, new_dotproduct
from tensor_buffer import Tensor

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
      self.layer1 = Layer(Tensor(np.array([0.1, 0.5])).data, 2, 0.25)
      self.weights1 = self.layer1.get_weights()
      self.layer2 = Layer(Tensor(np.array([1,2])).data, 2, 0.35)
   
   def forward(self, layer):
      inp = layer.inp
      weights = layer.get_weights()

      w1 = []
      for i in range(0, len(weights) - 2):
         if i == 0: w1.append(weights[i])
         else: w1.append(weights[i + 1])
      w2 = []
      for i in range(1, len(weights) - 1):
         if i == 1: w2.append(weights[i])
         else: w2.append(weights[i + 1])
      
      h1 = new_dotproduct(inp, w1) + layer.bias
      output1 = Sigmoid.forward(h1)
      h2 = new_dotproduct(inp, w2) + layer.bias
      output2 = Sigmoid.forward(h2)
      print(output1, output2)

nn = Model()
nn.forward(nn.layer1)

w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
w1 = []
for i in range(0,2):
   if i == 0: w1.append(w[i]) 
   else: w1.append(w[i + 1])
#print(w1)
w2 = []
for i in range(1,3):
   if i == 1: w2.append(w[i]) 
   else: w2.append(w[i + 1])
#print(w2)
w3 = []
for i in range(4,6):
   if i == 4: w3.append(w[i]) 
   else: w3.append(w[i + 1])
#print(w3)
w4 = []
for i in range(5,7):
   if i == 5: w4.append(w[i]) 
   else: w4.append(w[i + 1])
#print(w4)