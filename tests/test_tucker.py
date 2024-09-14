from source.decompositions.tucker import *
import pytest


class Testtucker:
    def setup_method(self, method):
        print(f"Setting up {method}")
        tmpcore = np.random.randn(4, 5, 3); tmp1 = np.random.randn(7, 4); tmp2 = np.random.randn(5, 5); tmp3 = np.random.randn(5, 3)
        self.X = tuckerrec(tmpcore, [tmp1, tmp2, tmp3])
        
    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def test_tuckerhosvd(self):
        c, f = tuckerhosvd(self.X, ranks=[7, 5, 5])
        res = tuckerrec(c, f)
        check = np.linalg.norm(self.X - res)/np.linalg.norm(self.X)
        assert check <= 1e-6

    def test_tuckerhooi(self):
        c, f = tuckerhooi(self.X, ranks=[7, 5, 5])
        res = tuckerrec(c, f)
        check = np.linalg.norm(self.X - res)/np.linalg.norm(self.X)
        assert check <= 1e-6