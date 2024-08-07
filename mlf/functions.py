from mlf.tensor import Function
import numpy as np

# TODO: make tests for when a function gets an undefined value as a result (Sqrt, Log, Div)


# Unary Fxns
class Negative(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.negative(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return -(grad)


class Reciprocal(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = np.array(x, dtype=np.float32)
        return np.reciprocal(self.x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return -(self.x**-2) * grad


class Sqrt(Function):  # needs to handle negative numbers
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.sqrt(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return (0.5) * (1 / self.x**0.5) * grad


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.ret = np.exp(x)
        return self.ret

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.ret * grad


class Log(Function):  # ln(x) # needs to handle negative numbers and 0
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.log(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return 1 / self.x * grad


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.sin(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.cos(self.x) * grad


class Relu(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = np.maximum(x, 0)
        return self.x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(self.x >= 0, grad, self.x)


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        denom = np.add(1, np.exp(-x))
        self.x = np.divide(1, denom)
        return self.x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.x * (1 - self.x) * grad


# Binary Fxns
class Add(Function):
    def forward(self, x: np.ndarray, y) -> np.ndarray:
        return np.add(x, y)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad if self.needs_grad[0] else None, grad if self.needs_grad[1] else None


class Sub(Function):
    def forward(self, x: np.ndarray, y) -> np.ndarray:
        return x + -(y)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad if self.needs_grad[0] else None, -grad if self.needs_grad[1] else None


class Mul(Function):
    def forward(self, x: np.ndarray, y) -> np.ndarray:
        self.x, self.y = x, y
        return np.multiply(x, y)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.y * grad if self.needs_grad[0] else None, self.x * grad if self.needs_grad[1] else None


class Div(Function):  # needs to handle division by 0
    def forward(self, x: np.ndarray, y) -> np.ndarray:
        self.x, self.y = x, y
        return np.divide(x, y)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return (1 / self.y) * grad if self.needs_grad[0] else None, (-self.x * self.y**-2) * grad if self.needs_grad[
            1
        ] else None


# Movement Fxns
class Reshape(Function):
    def forward(self, x: np.ndarray, shape) -> np.ndarray:
        self.input_shape = x.shape
        return np.reshape(x, shape)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.reshape(grad, self.input_shape)


class Permute(Function):
    def forward(self, x: np.ndarray, axis) -> np.ndarray:
        self.input_axis = axis
        return np.permute_dims(x, axis)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # how does np.argsort work?
        return np.permute_dims(grad, np.argsort(self.input_axis))


class Flip(Function):
    def forward(self, x: np.ndarray, axis) -> np.ndarray:
        self.axis = axis
        return np.flip(x, axis)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.flip(grad, axis=self.axis)


class Pad(Function):
    def forward(self, x: np.ndarray, width, mode, **kwargs) -> np.ndarray:
        self.originals = tuple((p[0], s + p[0]) for p, s in zip(width, x.shape))
        return np.pad(x, width, mode, **kwargs)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        slices = tuple(slice(i[0], i[1]) for i in self.originals)
        return grad[slices]


class Shrink(Function):
    def forward(self, x, axis):
        return np.squeeze(x, axis)  # incorrect


class Expand(Function):
    def forward(self, x: np.ndarray, newshape) -> np.ndarray:
        self.axis = tuple(i for i, (s1, s2) in enumerate(zip(x.shape, newshape)) if s1 != s2)
        return np.broadcast_to(x, newshape)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.sum(axis=self.axis, keepdims=True)


# Reduce Fxns
class Sum(Function):
    def forward(self, x: np.ndarray, axis, keepdims) -> np.ndarray:
        self.input_shape = np.array(x).shape
        return np.sum(x, axis, keepdims=keepdims)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.broadcast_to(grad, self.input_shape)
