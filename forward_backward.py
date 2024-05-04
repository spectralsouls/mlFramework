import numpy as np
from mlops import Sigmoid, MSE

# This needs to work with batches toos
# Add a Sigmoid layer

class DenseLayer:
   def __init__(self, num_inp, num_nodes):
      self.num_inp, self.num_nodes = num_inp, num_nodes
      self.weights = np.empty(shape=(num_inp, num_nodes))
      self.bias = np.empty([0])
   
   def forward(self, inp):
      self.output = np.matmul(inp, self.weights) + self.bias #check if a@b is faster than np.matmul(a,b)

   def backward(self, d_inp):
      pass

fc1 = DenseLayer(2,2)
fc1.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
fc1.bias = np.array([0.25])

fc2 = DenseLayer(2,2)
fc2.weights = np.array([[0.5,0.7],[0.6,0.8]])
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
     d_MSE = MSE.backward(test_val, pred)
     d_act2 = self.sigmoid.backward(self.layer2.output)
     self.layer2.backward(deriv)
     print(f"d_MSE:{d_MSE}, d_act2: {d_act2}, d_layer2: {self.sigmoid.forward(self.layer1.output)}")
     d_weight = d_MSE[0] * d_act2[0] * self.sigmoid.forward(self.layer1.output) # self.sigmoid.forward(self.layer1.output) --> should not have to calculate this again during backwards pass
     print(d_weight)
     return d_weight
   
   def update_weights(self, learning_rate):
      pass


# FORWARD PASS
inp = np.array([0.1, 0.5])
batch_inp = np.random.rand(3,2)
nn = Model()
pred = nn.forward(inp)

# LOSS
test_val = np.array([0.05, 0.95])
error = MSE.forward(test_val, pred)
d_error = MSE.backward(test_val, pred)

# BACKWARD PASS
update = nn.backward(pred)
print(update)

print(f"pred:{pred}, error:{error}")


