from tensor import Function
import numpy as np

# Unary Fxns
class Negative(Function):
    def forward(x): return np.negative(x)

class Reciprocal(Function):
    def forward(x): return np.reciprocal(x)

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
    def forward(x): pass # ----> Compute Sigmoid Function

# Binary Fxns