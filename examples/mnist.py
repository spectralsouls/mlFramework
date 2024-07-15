import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import functools

torch.manual_seed(42)

def sequential(inp, layers):
    return functools.reduce(lambda x,f: f(x), layers, inp)

class Model(nn.Module):
    def __init__(self):
        self.layers = [
            nn.Conv2d(1, 32, 3, 1), F.relu,
            nn.Conv2d(32, 64, 3, 1), F.relu,
            lambda x: F.max_pool2d(x, kernel_size=(2,2)), nn.Dropout(0.25),
            torch.flatten, nn.Linear(9216, 128),
            F.relu, nn.Dropout(0.5), 
            nn.Linear(128, 10), 
            lambda x: F.log_softmax(x, dim=0)
                      ]
            
    def __call__(self, x): return sequential(x, self.layers)
        
model = Model()
inp = torch.rand(size=[1, 28, 28])

a = sequential(inp, model.layers)
print(a)

