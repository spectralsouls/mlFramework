import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import functools


torch.manual_seed(42)

def sequential(inp, layers):
    return functools.reduce(lambda x,f: f(x), layers, inp)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = [
            nn.Conv2d(1, 32, 5), F.relu,
            nn.Conv2d(32, 32, 5), F.relu,
            nn.BatchNorm2d(32), lambda x: F.max_pool2d(x, (2,2)), 
            nn.Conv2d(32, 64, 3), F.relu, 
            nn.Conv2d(64, 64, 3), F.relu,
            nn.BatchNorm2d(64), lambda x: F.max_pool2d(x, (2,2)),
            lambda x: torch.flatten(x, start_dim=1), nn.Linear(576, 10)
                      ]
                 
    def __call__(self, x): return sequential(x, self.layers)

model = Model()
#adam = optim.Adam(model.parameters)

def train_step(bs=512):
    #adam.zero_grad()
    samples = torch.randint(low=0, high=60000, size=(bs,)) 
    loss = F.cross_entropy(model(x_train[samples].float()), y_train[samples])#.backward()
    print(f"loss: {loss: 4.2f}")
    loss.backward()
    #adam.step()
    