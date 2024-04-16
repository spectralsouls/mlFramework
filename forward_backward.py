import numpy as np
from mlops import Sigmoid, dotproduct
from tensor_buffer import Tensor

class DenseLayer:
   def __init__(self, num_inp, num_nodes):
      self.num_inp = num_inp
      self.num_nodes = num_nodes
      self.weights = [] # add random initialization later
      self.bias = [0]  # should this always start with an initial value of zero?    
   
   def forward(self, inp):
      self.output = dotproduct(inp, self.weights) + self.bias

fc1 = DenseLayer(2,2)
fc1.weights.append([0.1,0.2,0.3,0.4])

fc2 = DenseLayer(2,2)
fc2.weights.append([0.5,0.6,0.7,0.8])


class Model:
   def __init__(self):
      self.layer1 = fc1
      self.layer2 = fc2

   def forward(self, inp):
      out = []
      out.append(self.layer1.forward(inp))
      out.append(self.layer2.forward(self.layer1.output))  

      w1 = []
      for i in range(0, len(weights) - 2):
         if i == 0: w1.append(weights[i])
         else: w1.append(weights[i + 1])
      w2 = []
      for i in range(1, len(weights) - 1):
         if i == 1: w2.append(weights[i])
         else: w2.append(weights[i + 1])
      
      h1 = dotproduct(inp, w1) + layer.bias
      output1 = Sigmoid.forward(h1)
      h2 = dotproduct(inp, w2) + layer.bias
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