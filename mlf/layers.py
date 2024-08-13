from mlf.tensor import tensor


class Linear:
    def __init__(self, inp_features, out_features, bias=True):
        self.weights = tensor.random((inp_features, out_features), requires_grad=True)
        self.bias = tensor.random(out_features, requires_grad=True) if bias else None

    def __call__(self, inp: tensor):
        return inp.linear(self.weights, self.bias)
