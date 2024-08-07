from tensor import tensor

class Linear:
    def __init__(self, inp_features, out_features, bias=True):
        self.weights = tensor.random(inp_features, out_features)
        self.bias = tensor.random(out_features) if bias else None

    def __call__(self, inp:tensor): return inp.linear(self.weights, self.bias)