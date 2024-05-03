import numpy as np
from mlops import Sigmoid, MSE

# This needs to work with batches toos

class DenseLayer:
   def __init__(self, num_inp, num_nodes):
      self.num_inp, self.num_nodes = num_inp, num_nodes
      self.weights = np.empty(shape=(num_inp, num_nodes))
      self.bias = np.empty([0])
   
   def forward(self, inp):
      self.output = np.dot(self.weights, inp) + self.bias

fc1 = DenseLayer(2,2)
fc1.weights = np.array([[0.1, 0.3], [0.2, 0.4]])
fc1.bias = np.array([0.25])

fc2 = DenseLayer(2,2)
fc2.weights = np.array([[0.5,0.6],[0.7,0.8]])
fc2.bias = np.array([0.35])

act = Sigmoid

class Model:
   def __init__(self):
      self.layer1 = fc1
      self.layer2 = fc2
      self.sigmoid = act

   def forward(self, inp):
      self.layer1.forward(inp)
      act1 = self.sigmoid.forward(self.layer1.output)
      self.layer2.forward(act1)  
      out = self.sigmoid.forward(self.layer2.output)
      return out
   
   def backward(self, deriv):
     dact1 = self.sigmoid.backward(deriv)
     return dact1


# FORWARD PASS
inp = np.array([0.1, 0.5])
batch_inp = np.random.rand(3,2)
nn = Model()
pred = nn.forward(inp)
print(pred)

test_val = np.array([0.05, 0.95])
error = MSE.forward(test_val, pred)
print(error)

# BACKWARD PASS
derivatives = 0.73492
update = nn.backward(derivatives)
#print(update)


