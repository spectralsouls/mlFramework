import unittest
import torch
import numpy as np
from tensor import tensor

default_vals = [-1,0,2]
test_vals = [-1,0,2]


def prepare_tensors(val):
    pytorch = torch.tensor(data=val)
    native = tensor(data=val)
    return pytorch, native

def perform_test(val, torch_fxn, native_fxn=None):
    if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(val)
    torch_result, native_result = torch_fxn(pytorch_tensor), native_fxn(native_tensor)
   # print(f"Result: {torch_result}")
    np.testing.assert_allclose(torch_result.numpy(), native_result.numpy())

def perform_binop_test(load_vals,test_vals, torch_fxn, native_fxn=None):
    if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(load_vals)
    torch_test_vals, native_test_vals = torch.tensor(test_vals), tensor(test_vals)
    torch_result, native_result = torch_fxn(pytorch_tensor, torch_test_vals), native_fxn(native_tensor, native_test_vals)
   # print(f"Result: {torch_result}")
    np.testing.assert_allclose(torch_result.numpy(), native_result.numpy())



class TestUnaryOps(unittest.TestCase):
    def test_neg(self): perform_test(default_vals, lambda x: x.negative())
    def test_recip(self): perform_test(default_vals, torch.reciprocal, tensor.recip) # Runtime warning
    def test_sqrt(self): perform_test(default_vals, lambda x: x.sqrt()) # Runtime warning
    def test_exp(self): perform_test(default_vals, lambda x: x.exp())
    def test_log(self): perform_test(default_vals, lambda x: x.log())
    def test_sin(self): perform_test(default_vals, lambda x: x.sin())
    def test_relu(self): perform_test(default_vals, lambda x: x.relu())

class TestBinaryOps(unittest.TestCase):
    def test_add_constant(self): perform_binop_test(default_vals, test_vals, lambda x,y: x + y)
    def test_sub(self): perform_binop_test(default_vals, test_vals, lambda x,y: x - y)
    def test_mul(self): perform_binop_test(default_vals, test_vals, lambda x,y: x * y)
    def test_div(self): perform_binop_test(default_vals, test_vals, lambda x,y: x / y)


if __name__ == '__main__':
    unittest.main()
