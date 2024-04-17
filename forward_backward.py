import numpy as np
from mlops import Sigmoid, dotproduct
from tensor_buffer import Tensor

class DenseLayer:
   def __init__(self, num_inp, num_nodes):
      self.num_inp = num_inp
      self.num_nodes = num_nodes
      self.weights = np.empty(shape=(num_inp, num_nodes)) # add random initialization later
      self.bias = Tensor([0])  # should this always start with an initial value of zero?    
   
   def forward(self, inp):
      self.output = Tensor(dotproduct(inp, self.weights)) + self.bias # --> you shouldnt have to wrap this in a Tensor

fc1 = DenseLayer(2,2)
fc1.weights = Tensor([[0.1, 0.3], [0.2, 0.4]])

fc2 = DenseLayer(2,2)
fc2.weights = Tensor([[0.5,0.7],[0.6,0.8]])

class Model:
   def __init__(self):
      self.layer1 = fc1
      self.layer2 = fc2

   def forward(self, inp):
      self.layer1.forward(inp)
      self.layer2.forward(self.layer1.output)  
      return self.layer2.output


inp = Tensor([0.1, 0.5])
nn = Model()
out = nn.forward(inp)
print(out.data)
