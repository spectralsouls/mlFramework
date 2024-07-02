import unittest
import torch
import numpy as np
from tensor import tensor

def prepare_tensors(val):
    pytorch = torch.tensor(data=val)
    native = tensor(data=val)
    return pytorch, native

def prepare_test(val, torch_fxn, native_fxn=None):
    if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(val)
    torch_result = torch_fxn(pytorch_tensor)
    native_result = native_fxn(native_tensor)
    print(f"Result: {torch_result}")
    np.testing.assert_allclose(torch_result.numpy(), native_result)

class TestUnaryOps(unittest.TestCase):
    def test_exp(self): prepare_test(3, torch.exp, tensor.exp)
    def test_log(self): prepare_test(4, torch.log, tensor.log)

if __name__ == '__main__':
    unittest.main()
