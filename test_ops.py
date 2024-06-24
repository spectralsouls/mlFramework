import unittest
import torch
import numpy as np
from tinygrad.tensor import Tensor

def prepare_tensors(val):
    pytorch = torch.tensor(data=val)
    tinygrad = Tensor(data=val)
    return pytorch, tinygrad

def prepare_test(val, torch_fxn, tinygrad_fxn=None):
    if tinygrad_fxn == None: tinygrad_fxn = torch_fxn
    pytorch_tensor, tinygrad_tensor = prepare_tensors(val)
    torch_result = torch_fxn(pytorch_tensor)
    tinygrad_result = tinygrad_fxn(tinygrad_tensor)
    print(f"Result: {torch_result}")
    np.testing.assert_allclose(torch_result.numpy(), tinygrad_result.numpy())

class TestOps(unittest.TestCase):
    def test_exp(self): prepare_test(3, torch.exp, Tensor.exp)
    def test_log(self): prepare_test(4, torch.log, Tensor.log)

if __name__ == '__main__':
    unittest.main()
