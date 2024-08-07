import unittest
from tensor import tensor
import numpy as np


# Test data
class TestData(unittest.TestCase):
    def test_constant(self):
        x = tensor(3)
        assert x.data == 3

    def test_tuple(self):
        x = tensor((-1, 2))
        assert x[0] == -1
        assert x[-1] == 2

    def test_list(self):
        x = tensor([-1, 0, 2])
        assert x[0] == -1
        assert x[-2] == 0
        assert x[2] == 2

    def test_nested_list(self):
        x = tensor([[1, -1], [0, 2]])
        assert x[0][0] == 1
        assert x[1][1] == 2
        assert x[-1][-2] == 0

    def test_numpy(self):
        x = tensor(np.array(np.arange(20)).reshape(4, 5))
        assert x.shape == (4, 5)
        assert np.all(x[0] == [0, 1, 2, 3, 4])
        assert x[3][4] == 19
        assert np.all(x[-1] == [15, 16, 17, 18, 19])
        y = tensor(np.array([1, 2, 3, 4]))
        assert y.shape == (4,)
        assert np.all(y.data == [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
