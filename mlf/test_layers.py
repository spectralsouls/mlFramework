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

        inp = tensor.random(inp_size)
        t_inp = torch.tensor(inp.numpy())
        #w = np.random.uniform(size=(inp_size,out_size))
        #b = np.random.uniform(size=out_size)

        native_layer = mlf.Linear(inp_size, out_size)
        torch_layer = nn.Linear(inp_size, out_size)

        torch_layer.weight = nn.Parameter(torch.tensor(native_layer.weights.transpose().numpy(), requires_grad=True))
        torch_layer.bias = nn.Parameter(torch.tensor(native_layer.bias.numpy(), requires_grad=True))
        
        native_result = native_layer(inp)
        torch_result = torch_layer(t_inp)

        np.testing.assert_allclose(torch_result.detach().numpy(), native_result.numpy())

    def test_batchnorm2d(self):
        inp_shape = (2,3,2,2)
        inp = tensor.random(inp_shape)
        t_inp = torch.tensor(inp.numpy())

        native_layer = mlf.BatchNorm(3)
        torch_layer = nn.BatchNorm2d(3)

        native_result, torch_result = native_layer(inp), torch_layer(t_inp)

        # need to set the rtol, atol manually due to floating point imprecision, passes without manual tuning when using float64
        np.testing.assert_allclose(torch_result.detach().numpy(), native_result.numpy(), rtol=1e-6, atol=1e-6) 

if __name__ == "__main__":
    unittest.main()





