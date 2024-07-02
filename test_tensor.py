import unittest
from tensor import tensor

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
        x = tensor([[1, -1], [0 , 2]])
        assert x[0][0] == 1
        assert x[1][1] == 2
        assert x[-1][-2] == 0


if __name__ == '__main__':
    unittest.main()