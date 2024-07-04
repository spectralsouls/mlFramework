import unittest
import torch
import numpy as np
from tensor import tensor

def prepare_tensors(val):
    pytorch = torch.tensor(data=val)
    native = tensor(data=val)
    return pytorch, native

def perform_test(val, torch_fxn, native_fxn=None):
    if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(val)
    torch_result = torch_fxn(pytorch_tensor)
    native_result = native_fxn(native_tensor)
    print(f"Result: {torch_result}")
    np.testing.assert_allclose(torch_result.numpy(), native_result)

class TestUnaryOps(unittest.TestCase):
    def test_neg(self): perform_test([-1,0,2], torch.negative, tensor.negative)
    def test_recip(self): perform_test([-1,0,2],torch.reciprocal, tensor.reciprocal) # Runtime warning
    def test_sqrt(self): perform_test([-1,0,2],torch.sqrt, tensor.sqrt) # Runtime warning
    def test_exp(self): perform_test(3, torch.exp, tensor.exp)
    def test_log(self): perform_test(4, torch.log, tensor.log)
    def test_sin(self): perform_test([-1,0,2],torch.sin, tensor.sin)
    def test_relu(self): perform_test((3, -1), torch.relu, tensor.relu)

if __name__ == '__main__':
    unittest.main()
