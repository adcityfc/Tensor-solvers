from source.decompositions.tt import *
import pytest


class Testtt:
    def setup_method(self, method):
        print(f"Setting up {method}")
        # Create a tensor
        tmp1 = np.random.rand(1, 5, 5); tmp2 = np.random.rand(5, 3, 8); tmp3 = np.random.rand(8, 7, 1)
        self.X = ttrec([tmp1, tmp2, tmp3])


    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def test_tt(self):
        factors = ttsvd(self.X, rank=[1, 5, 8, 1])
        res = ttrec(factors)
        check = np.linalg.norm(res - self.X)/np.linalg.norm(self.X)
        print(check)
        assert check <= 1e-6