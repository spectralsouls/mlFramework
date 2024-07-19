from tensor import Function
import numpy as np

# Unary Fxns
class Negative(Function):
    def forward(self, x): return np.negative(x)

    def backward(self, grad): return -(grad)

class Reciprocal(Function):
    def forward(self, x): 
        self.x = np.array(x, dtype=np.float32)
        return np.reciprocal(self.x)
    
    def backward(self, grad): return -(self.x**-2) * grad

class Sqrt(Function):
    def forward(self, x): 
        self.x = x
        return np.sqrt(x)

    def backward(self, grad): 
        return (0.5) * (1/self.x**0.5) * grad

class Exp(Function):
    def forward(self, x): 
        self.ret = np.exp(x)
        return self.ret

    def backward(self, grad): return self.ret * grad

class Log(Function): # ln(x)
    def forward(self, x): 
        self.x = x
        return np.log(x)
    
    def backward(self, grad): return 1/self.x * grad

class Sin(Function):
    def forward(self, x): 
        self.x = x
        return np.sin(x)
    
    def backward(self, grad):
        return np.cos(self.x) * grad

class Relu(Function):
    def forward(self, x): 
        self.x = np.maximum(x, 0)
        return self.x

    def backward(self, grad): 
        return grad if self.x >= 1 else 0

class Sigmoid(Function):
    def forward(self, x): 
        denom = np.add(1, np.exp(-x))
        self.x = np.divide(1, denom)
        return self.x
    
    def backward(self, grad): 
        return self.x * (1 - self.x) * grad

# Binary Fxns
class Add(Function):
    def forward(self, x, y): return np.add(x,y) 

    def backward(self, grad): 
        return grad, grad

class Sub(Function):
    def forward(self, x, y): return x + -(y)

    def backward(self, grad): 
        return grad, -grad

class Mul(Function):
    def forward(self, x, y): 
        self.x, self.y = x, y
        return np.multiply(x, y)
    
    def backward(self, grad):
        return self.y * grad, self.x * grad
    
class Div(Function):
    def forward(self, x, y):
        self.x, self.y = x, y
        return np.divide(x, y)
    
    def backward(self, grad):
        return (1/self.y) * grad, (-self.x*self.y**-2) * grad
    

# Movement Fxns
class Reshape(Function):
    def forward(self, x, shape:tuple): return np.reshape(x, shape)
