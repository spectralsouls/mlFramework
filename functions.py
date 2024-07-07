from tensor import Function
import numpy as np

# Unary Fxns
class Negative(Function):
    def forward(x): return np.negative(x)

class Reciprocal(Function):
    def forward(x): 
        x_casted = np.array(x, dtype=np.float32)
        return np.reciprocal(x_casted)

class Sqrt(Function):
    def forward(x): return np.sqrt(x)

class Exp(Function):
    def forward(x): return np.exp(x)

class Log(Function):
    def forward(x): return np.log(x)

class Sin(Function):
    def forward(x): return np.sin(x)

class Relu(Function):
    def forward(x): return np.maximum(x, 0)

class Sigmoid(Function):
    def forward(x): 
        denom = np.add(1, np.exp(np.negative(x)))
        return np.divide(1, denom)

# Binary Fxns
class Add(Function):
    def forward(x, y): return np.add(x,y) 

class Mul(Function):
    def forward(x, y): return np.multiply(x, y)