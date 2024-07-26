import unittest
import torch
import numpy as np
from tensor import tensor


# make a FORWARD_ONLY env variable
#default_vals = [-1,0,2]
#test_vals = [-1,0,2]
default_vals = 2.0
test_vals = 3.0


def prepare_tensors(shape, forward_only):
    np.random.seed(0)
    data = [np.random.uniform(size=s) for s in shape]
    pytorch = [torch.tensor(data=d, requires_grad=(not forward_only), dtype=torch.float64) for d in data] # should change to float32 later
    native = [tensor(data=d.detach().numpy(), requires_grad=(not forward_only)) for d in pytorch]
    return pytorch, native

def compare(msg, torch, native):
    #print(f"torch:{torch}, native:{native}")
    print(msg)
    np.testing.assert_allclose(torch.numpy(), native.numpy())

def perform_test(shape, torch_fxn, native_fxn=None, forward_only=False):
    if native_fxn == None: native_fxn = torch_fxn
    pytorch_tensor, native_tensor = prepare_tensors(shape, forward_only)
    torch_result, native_result = torch_fxn(*pytorch_tensor), native_fxn(*native_tensor)
    #print(f"Result: {torch_result}")
    compare("FORWARD", torch_result.detach(), native_result)

    if not forward_only:
        torch_result.mean().backward(), native_result.mean().backwards() # TODO: x.square().mean().backwards() doesnt work
        for p, n in zip(pytorch_tensor, native_tensor):
            compare("BACKWARD", p.grad.detach(), n.grad)

class TestUnaryOps(unittest.TestCase):
    def test_neg(self): 
        perform_test([(4,5)], lambda x: -x)
        perform_test([(4,5)], lambda x: x.negative())
        perform_test([()], lambda x: x.negative())
    def test_recip(self): perform_test([default_vals], torch.reciprocal, tensor.recip) # Runtime warning
    def test_sqrt(self): perform_test([default_vals], lambda x: x.sqrt()) # Runtime warning
    def test_exp(self): perform_test([default_vals], lambda x: x.exp())
    def test_log(self): perform_test([default_vals], lambda x: x.log())
    def test_sin(self): perform_test([default_vals], lambda x: x.sin())
    def test_relu(self): perform_test([default_vals], lambda x: x.relu())
    def test_sigmoid(self): perform_test([default_vals], lambda x: x.sigmoid()) 

class TestBinaryOps(unittest.TestCase):
    def test_add(self): perform_test([default_vals, test_vals], lambda x,y: x + y)
    def test_sub(self): perform_test([default_vals, test_vals], lambda x,y: x - y)
    def test_mul(self): perform_test([default_vals, test_vals], lambda x,y: x * y)
    def test_div(self): perform_test([default_vals, test_vals], lambda x,y: x / y)


class TestMovementOps(unittest.TestCase):
    def test_reshape(self): pass

if __name__ == '__main__':
    unittest.main()
