import unittest
import torch
import numpy as np
from tensor import tensor


# make a FORWARD_ONLY env variable
#default_vals = [-1,0,2]
#test_vals = [-1,0,2]
default_vals = 2.0
test_vals = 3.0


def prepare_tensors(val, forward_only):
    pytorch = torch.tensor(data=val, requires_grad=(not forward_only), dtype=torch.float64) # should change to float32 later
    native = tensor(data=val)
    return pytorch, native

def perform_test(val, torch_fxn, native_fxn=None, forward_only=False):
    if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(val, forward_only)
    torch_result, native_result = torch_fxn(pytorch_tensor), native_fxn(native_tensor)
    #print(f"Result: {torch_result}")
    np.testing.assert_allclose(torch_result.detach().numpy(), native_result.numpy())

    if not forward_only:
        #print("BACKWARD_PASS")
        torch_result.backward(), native_result.backwards()
        #print(pytorch_tensor.grad, native_tensor.grad)
        np.testing.assert_allclose(pytorch_tensor.grad.numpy(), native_tensor.grad.numpy())


def perform_binop_test(load_vals,test_vals, torch_fxn, native_fxn=None, forward_only=False):
    if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(load_vals, forward_only)
    torch_test_vals, native_test_vals = prepare_tensors(test_vals, forward_only)
    torch_result, native_result = torch_fxn(pytorch_tensor, torch_test_vals), native_fxn(native_tensor, native_test_vals)
   # print(f"Result: {torch_result}")
    np.testing.assert_allclose(torch_result.detach().numpy(), native_result.numpy())

    if not forward_only:
        torch_result.backward(), native_result.backwards()
        np.testing.assert_allclose(pytorch_tensor.grad.numpy(), native_tensor.grad.numpy())
        np.testing.assert_allclose(torch_test_vals.grad.numpy(), native_test_vals.grad.numpy())
        



class TestUnaryOps(unittest.TestCase):
    def test_neg(self): perform_test(default_vals, lambda x: x.negative())
    def test_recip(self): perform_test(default_vals, torch.reciprocal, tensor.recip) # Runtime warning
    def test_sqrt(self): perform_test(default_vals, lambda x: x.sqrt()) # Runtime warning
    def test_exp(self): perform_test(default_vals, lambda x: x.exp())
    def test_log(self): perform_test(default_vals, lambda x: x.log())
    def test_sin(self): perform_test(default_vals, lambda x: x.sin())
    def test_relu(self): perform_test(default_vals, lambda x: x.relu())
    def test_sigmoid(self): perform_test(default_vals, lambda x: x.sigmoid()) 

class TestBinaryOps(unittest.TestCase):
    def test_add(self): perform_binop_test(default_vals, test_vals, lambda x,y: x + y)
    def test_sub(self): perform_binop_test(default_vals, test_vals, lambda x,y: x - y)
    def test_mul(self): perform_binop_test(default_vals, test_vals, lambda x,y: x * y)
    def test_div(self): perform_binop_test(default_vals, test_vals, lambda x,y: x / y)


if __name__ == '__main__':
    unittest.main()
