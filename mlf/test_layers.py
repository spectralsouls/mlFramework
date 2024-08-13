import unittest
import mlf.layers as mlf
import torch.nn as nn
import numpy as np
import torch
from mlf.tensor import tensor

np.random.seed(0)

class TestLayers(unittest.TestCase):
    def test_linear(self):
        inp_size, out_size = 4, 2

        inp = np.random.uniform(size=inp_size)
        #w = np.random.uniform(size=(inp_size,out_size))
        #b = np.random.uniform(size=out_size)
        
        a = tensor(inp)
        t = torch.tensor(a.numpy())

        native = mlf.Linear(inp_size, out_size)
        torch_layer = nn.Linear(inp_size, out_size)

        torch_layer.weight = nn.Parameter(torch.tensor(native.weights.transpose().numpy(), requires_grad=True))
        torch_layer.bias = nn.Parameter(torch.tensor(native.bias.numpy(), requires_grad=True))
        
        native_result = native(a)
        torch_result = torch_layer(t)

        np.testing.assert_allclose(torch_result.detach().numpy(), native_result.numpy())

if __name__ == "__main__":
    unittest.main()





