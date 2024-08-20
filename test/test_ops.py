import unittest
import torch
import numpy as np
from mlf.tensor import tensor
import math

# make a FORWARD_ONLY env variable
# create tests for mean, linear and batchnorm,

def prepare_tensors(shape, forward_only, vals):
    np.random.seed(0)
    data = vals if vals is not None else [np.random.uniform(size=s) for s in shape]
    pytorch = [
        torch.tensor(data=d, requires_grad=(not forward_only)) for d in data
    ]
    native = [tensor(data=d.detach().numpy(), requires_grad=(not forward_only)) for d in pytorch]
    return pytorch, native


def compare(msg, torch, native):
    # print(f"torch:{torch}, native:{native}")
    print(msg)
    # need to set the rtol, atol manually due to floating point imprecision, passes without manual tuning when using float64
    np.testing.assert_allclose(torch.numpy(), native.numpy(), atol=1e-6)
#rtol=1e-6, atol=1e-6

def perform_test(shape, torch_fxn, native_fxn=None, forward_only=False, vals=None):
    native_fxn = torch_fxn if native_fxn is None else native_fxn
    pytorch_tensor, native_tensor = prepare_tensors(shape, forward_only, vals)
    torch_result, native_result = torch_fxn(*pytorch_tensor), native_fxn(*native_tensor)
    # print(f"Result: {torch_result}")
    compare("FORWARD", torch_result.detach(), native_result)
    if not forward_only:
        (
            torch_result.square().mean().backward(),
            native_result.square().mean().backwards(),
        )
        for p, n in zip(pytorch_tensor, native_tensor):
            compare("BACKWARD", p.grad.detach(), n.grad)


class TestUnaryOps(unittest.TestCase):
    def test_neg(self):
        print("***NEGATIVE***")
        perform_test([(4, 5)], lambda x: x.negative())
        perform_test([(4, 5)], lambda x: -x)
        perform_test([()], lambda x: x.negative())

    # def test_recip(self): perform_test([default_vals], torch.reciprocal, tensor.recip)
    def test_sqrt(self):
        print("***SQUARE ROOT***")
        perform_test([(4, 5)], lambda x: x.sqrt())
        perform_test([()], lambda x: x.sqrt())
        # perform_test([()], lambda x: x.sqrt(), vals=[-1])

    def test_exp(self):
        print("***e^x***")
        perform_test([(4, 5)], lambda x: x.exp())
        perform_test([()], lambda x: x.exp())

    def test_log(self):
        print("***ln(x)***")
        perform_test([(4, 5)], lambda x: x.log())
        perform_test([()], lambda x: x.log())
        # perform_test([()], lambda x: x.log(), vals=[0])
        # perform_test([()], lambda x: x.log(), vals=[-1])

    def test_sin(self):
        print("***SINE***")
        perform_test([(4, 5)], lambda x: x.sin())
        perform_test([()], lambda x: x.sin())

    def test_relu(self):
        print("***RELU***")
        perform_test([(4, 5)], lambda x: x.relu())
        perform_test([()], lambda x: x.relu())
        perform_test([(1, 2)], lambda x: x.relu(), vals=[[1.0, 2.0, -3.0, -4.0, 5.0, 0.0]])

    def test_sigmoid(self):
        print("***SIGMOID***")
        perform_test([(4, 5)], lambda x: x.sigmoid())
        perform_test([()], lambda x: x.sigmoid())


class TestBinaryOps(unittest.TestCase):
    def test_add(self):
        print("***ADDITION***")
        perform_test([(4, 5), (4, 5)], lambda x, y: x + y, tensor.add)  # check this one
        perform_test([(4, 5), (4, 5)], lambda x, y: x + y)
        perform_test([(), ()], lambda x, y: x + y)

    def test_sub(self):
        print("***SUBTRACT***")
        perform_test([(4, 5), (4, 5)], lambda x, y: x - y, tensor.sub)
        perform_test([(4, 5), (4, 5)], lambda x, y: x - y)
        perform_test([(), ()], lambda x, y: x - y)

    def test_mul(self):
        print("***MULTIPLY***")
        perform_test([(2), (2)], lambda x, y: x * y, tensor.mul)
        perform_test([(4, 5), (4, 5)], lambda x, y: x * y)
        perform_test([(), ()], lambda x, y: x * y)

    def test_div(self):
        print("***DIVIDE***")
        perform_test([(4, 5), (4, 5)], lambda x, y: x / y, tensor.div)
        perform_test([(4, 5), (4, 5)], lambda x, y: x / y)
        perform_test([(), ()], lambda x, y: x / y)
        # perform_test([(), ()], lambda x, y: x / y, vals=[1, 0])


class TestMovementOps(unittest.TestCase):
    # check tinygrad movement ops tests later
    def test_reshape(self):
        print("***RESHAPE***")
        perform_test([(4, 3, 6, 6)], lambda x: x.reshape((12, 6, 6)))
        perform_test([(4, 3, 6, 6)], lambda x: x.reshape((-1, 3, 6, 6)))
        perform_test([()], lambda x: x.reshape(()))
        perform_test([(1,)], lambda x: x.reshape(()))
        perform_test([()], lambda x: x.reshape((1,)))
        perform_test([()], lambda x: x.reshape((1, 1, 1)))

    def test_permute(self):
        print("***PERMUTE***")
        perform_test([(1, 2, 3, 4)], lambda x: x.permute((3, 0, 2, 1)))
        perform_test([(3, 4, 5, 6)], lambda x: x.permute((3, 2, 1, 0)))
        perform_test([()], lambda x: x.permute(()))

    def test_flip(self):
        print("***FLIP***")
        perform_test([(4, 3, 6, 6)], lambda x: x.flip((0,)))
        perform_test([(4, 3, 6, 6)], lambda x: x.flip((0, 1)))
        perform_test([(4, 3, 6, 6)], lambda x: x.flip((0, 1, 3)))
        perform_test([(4, 3, 6, 6)], lambda x: x.flip((3,)))
        perform_test([(4, 3, 6, 6)], lambda x: x.flip((0, 1, 3)).flip(0))
        perform_test([(4, 3, 6, 6)], lambda x: x.flip((-1,)))
        perform_test([()], lambda x: x.flip(()))
        perform_test([(1)], lambda x: x.flip(()))
        perform_test([(4, 3, 6, 6)], lambda x: x.flip(()))

    def test_pad(self):
        print("***PAD***")
        perform_test([(3, 3)], lambda x: torch.nn.functional.pad(x, (1, 2, 3, 4)), lambda x: x.pad(((3, 4), (1, 2))))
        perform_test(
            [(3, 3)],
            lambda x: torch.nn.functional.pad(x, (1, 2, 3, 4), value=5),
            lambda x: x.pad(((3, 4), (1, 2)), constant_values=5),
        )
        perform_test(
            [(3, 3)],
            lambda x: torch.nn.functional.pad(x, (1, 2, 3, 4), value=math.inf),
            lambda x: x.pad(((3, 4), (1, 2)), constant_values=math.inf),
        )  # Runtime Warning
        perform_test(
            [(3, 3)],
            lambda x: torch.nn.functional.pad(x, (1, 2, 3, 4), value=-math.inf),
            lambda x: x.pad(((3, 4), (1, 2)), constant_values=-math.inf),
        )  # Runtime Warning
        perform_test(
            [(3, 3)],
            lambda x: torch.nn.functional.pad(x, (0, 0, 3, 4), value=1),
            lambda x: x.pad(((3, 4), (0, 0)), constant_values=1),
        )
        perform_test(
            [(3, 3)],
            lambda x: torch.nn.functional.pad(x, (0, 0, 0, 0), value=1),
            lambda x: x.pad(((0, 0), (0, 0)), constant_values=1),
        )

    def test_expand(self):
        print("***EXPAND***")
        perform_test([(4, 3, 1, 6)], lambda x: x.expand((4, 3, 2, 6)))
        perform_test([(1, 1, 1, 1)], lambda x: x.expand((4, 3, 2, 6)))
        perform_test([(4, 3, 1, 6)], lambda x: x.expand((6, 1, 4, 3, 2, 6)))
        # perform_test([(4,3,1,6)], lambda x: x.expand((0,1,4,3,2,6)))
        # perform_test([(4,3,1,6)], lambda x: x.expand((4,3,0,6)))
        perform_test([()], lambda x: x.expand((4, 3, 2, 6)))
        perform_test([()], lambda x: x.expand([]))


class TestReduceOps(unittest.TestCase):
    def test_sum(self):
        perform_test([(45, 3)], lambda x: x.sum())
        perform_test([(3, 4, 5, 6)], lambda x: x.sum(axis=(3,)))
        perform_test([(3, 4, 5, 6)], lambda x: x.sum(axis=(1, 3)))
        perform_test([(3, 4, 5, 6)], lambda x: x.sum(axis=(0, 2)))
        perform_test([(3, 4, 5, 6)], lambda x: x.sum(axis=(1, 2)))
        perform_test([(3, 4, 5, 6)], lambda x: x.sum(axis=(1,)))
        perform_test([()], lambda x: x.sum())
        perform_test([()], lambda x: x.sum(0))
        perform_test([()], lambda x: x.sum(-1))
        perform_test([()], lambda x: x.sum(()))


if __name__ == "__main__":
    unittest.main()
