from mlf.tensor import tensor


class Linear:
    def __init__(self, inp_features: int, out_features: int, bias=True):
        self.weights = tensor.random((inp_features, out_features), requires_grad=True)
        self.bias = tensor.random(out_features, requires_grad=True) if bias else None

    def __call__(self, inp: tensor):
        return inp.linear(self.weights, self.bias)


# https://arxiv.org/abs/1502.03167v3
class BatchNorm:
    def __init__(self, num_channels: int, epsilon=1e-05):
        # weights is gamma, bias is beta 
        self.channels = num_channels
        self.weights = tensor.ones(num_channels, requires_grad=True) #default value of 1
        self.bias = tensor.zeroes(num_channels, requires_grad=True)  #default value of 0
        self.eps = epsilon

    def __call__(self, inp: tensor):
        assert len(inp.shape) == 4, f"input shape needs to be 4D, (currently {inp.shape})"
        mean_axis = tuple(x for x in range(len(inp.shape)) if x != 1)
        b_mean = inp.mean(axis=mean_axis, keepdims=True)
        b_var = inp.square().mean(axis=mean_axis, keepdims=True).expand(inp.shape) - b_mean.square()
        return inp.batchnorm(b_mean, b_var, self.eps)
