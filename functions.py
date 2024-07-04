from tensor import Function
import numpy as np

class Relu(Function):
    def forward(x): return np.maximum(x, 0)

class Exp(Function):
    def forward(x): return np.exp(x)

class Log(Function):
    def forward(x): return np.log(x)