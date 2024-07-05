import unittest
import torch
import numpy as np
from tensor import tensor

default_vals = [-1,0,2]

def prepare_tensors(val):
    pytorch = torch.tensor(data=val)
    native = tensor(data=val)
    return pytorch, native

def perform_test(val, torch_fxn, native_fxn):
   # if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(val)
    torch_result, native_result = torch_fxn(pytorch_tensor), native_fxn(native_tensor)
   # print(f"Result: {torch_result}")
    np.testing.assert_allclose(torch_result.numpy(), native_result)

def perform_binop_test(load_vals,test_vals torch_fxn, native_fxn=None):
    #test_vals = [-1,0,2]
    if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(load_vals)
    torch_result, native_result = torch_fxn(pytorch_tensor, test_vals), native_fxn(native_tensor, test_vals)
    np.testing.assert_allclose(torch_result.numpy(), native_result)



class TestUnaryOps(unittest.TestCase):
    def test_neg(self): perform_test(default_vals, torch.negative, tensor.negative)
    def test_recip(self): perform_test(default_vals,torch.reciprocal, tensor.recip) # Runtime warning
    def test_sqrt(self): perform_test(default_vals,torch.sqrt, tensor.sqrt) # Runtime warning
    def test_exp(self): perform_test(default_vals, torch.exp, tensor.exp)
    def test_log(self): perform_test(default_vals, torch.log, tensor.log)
    def test_sin(self): perform_test(default_vals,torch.sin, tensor.sin)
    def test_relu(self): perform_test(default_vals, torch.relu, tensor.relu)

class TestBinaryOps(unittest.TestCase):
    def test_add_constant(self): perform_binop_test(default_vals, 1, lambda x,y: x + y)
    def test_sub(self): pass
    def test_mul(self): pass
    def test_div(self): pass


if __name__ == '__main__':
    unittest.main()
