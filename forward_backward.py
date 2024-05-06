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

   def backward(self, deriv):
      self.dinput = (np.ones(shape=self.weights.shape) * deriv).T
      self.dweights = np.ones(shape=self.weights.shape) * deriv

fc1 = DenseLayer(2,2)
fc1.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
fc1.bias = np.array([0.25])

fc2 = DenseLayer(2,2)
fc2.weights = np.array([[0.5,0.7],[0.6,0.8]])
fc2.bias = np.array([0.35])

print(fc2.weights.T[0])

class Model:
   def __init__(self):
      self.layer1 = fc1
      self.layer2 = fc2
      self.sigmoid = Sigmoid

   def forward(self, inp):
      self.layer1.forward(inp)
      act1 = self.sigmoid.forward(self.layer1.output)
      self.layer2.forward(act1)  
      out = self.sigmoid.forward(self.layer2.output)
      return out
   
   def backward(self, pred, inp):
     d_MSE = MSE.backward(test_val, pred)
     d_act2 = self.sigmoid.backward(self.layer2.output)
     self.layer2.backward(self.sigmoid.forward(self.layer1.output)) # # self.sigmoid.forward(self.layer1.output) --> change this afterwards
     print(f"d_MSE:{d_MSE}, d_act2: {d_act2}, d_layer2: {self.layer2.dinput}")
     x = d_MSE * d_act2 * self.layer2.dinput
    # print(x)
     d_act1 = self.sigmoid.backward(self.layer1.output)
     self.layer2.backward(self.layer2.weights)
     self.layer1.backward(inp)
     self.layer2.backward(self.layer2.weights.T[0])
     print(f"d_act1: {d_act1}, d_layer1: {self.layer1.dinput}, d_layer2: {self.layer2.dweights}")
     y1 = d_MSE * d_act2 * self.layer2.dinput * d_act1 * self.layer2.dweights * self.layer1.dinput
     print(f"y1: {y1}")
     y2 = d_MSE * d_act2 * self.layer2.dinput * d_act1 * self.layer1.dweights
     y_total = y1 + y2
     #print(y_total)
   
   def update_weights(self, learning_rate, gradient):
      updated_weights = self.weights - learning_rate * gradient
      return updated_weights


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
derivs = nn.backward(pred, inp)
# update = nn.update_weights(0.6, derivs)

print(f"pred:{pred}, error:{error}")


# E1: [[w1, w2], [w3, w4]] --> [[0.5, 0.6], [0.5, 0.6]]
# E2: [[w1, w2], [w3, w4]] --> [[0.7, 0.8], [0.7, 0.8]]