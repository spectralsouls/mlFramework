import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import functools
import itertools

# reaches ~96% accuracy

transform = transforms.Compose([transforms.ToTensor])
train_mnist = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_mnist = datasets.MNIST('../data', train=False, download=True, transform=transform)

def transform_x(mnist):
    x = mnist.data
    return x.unsqueeze(1).float()

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
                 
    def get_parameters(self):
        params = [layer.parameters() for layer in self.layers if isinstance(layer, nn.Module)]
        p = itertools.chain(*params)
        return p

    def __call__(self, x): return sequential(x, self.layers)

if __name__ == "__main__":
    x_train, x_test = transform_x(train_mnist), transform_x(test_mnist)
    y_train, y_test = train_mnist.targets, test_mnist.targets

model = Model()
adam = optim.Adam(model.get_parameters())

def train_step(bs=512):
    adam.zero_grad()
    samples = torch.randint(low=0, high=60000, size=(bs,)) 
    loss = F.cross_entropy(model(x_train[samples].float()), y_train[samples])#cant do .backward() here
    loss_val = loss 
    loss.backward()
    adam.step()
    return loss_val
    
def calc_acc(): return torch.sum((model(x_test).argmax(axis=1) == y_test) / len(y_test))*100
    
for i in range(23):
    loss = train_step()
    if i%10 == 9:
        acc = calc_acc()
        print(f"loss: {loss: 4.2f}, accuracy: {acc: 4.2f}% ")

