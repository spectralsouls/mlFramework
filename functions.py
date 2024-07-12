from tensor import Function
import numpy as np

# Unary Fxns
class Negative(Function):
    def forward(self, x): return np.negative(x)

    def backward(self, grad): return -(grad)

class Reciprocal(Function):
    def forward(self, x): 
        x_casted = np.array(x, dtype=np.float32)
        return np.reciprocal(x_casted)

class Sqrt(Function):
    def forward(self, x): return np.sqrt(x)

    def backward(self, grad): 
        return (1/2) * (1/np.sqrt(grad)) 

class Exp(Function):
    def forward(self, x): return np.exp(x)

class Log(Function):
    def forward(self, x): return np.log(x)

class Sin(Function):
    def forward(self, x): return np.sin(x)

class Relu(Function):
    def forward(self, x): return np.maximum(x, 0)

class Sigmoid(Function):
    def forward(self, x): 
        denom = np.add(1, np.exp(np.negative(x)))
        return np.divide(1, denom)

# Binary Fxns
class Add(Function):
    def forward(self, x, y): return np.add(x,y) 

    def backward(self, grad): 
        return grad, grad

class Sub(Function):
    def forward(self, x, y): return x + -(y)

class Mul(Function):
    def forward(self, x, y): 
        self.x, self.y = x, y
        return np.multiply(x, y)
    
    def backward(self, grad):
        return self.y * grad, self.x * grad