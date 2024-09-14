from source.decompositions.cp import *
import pytest


class Testcp:
    def setup_method(self, method):
        print(f"Setting up {method}")
        tmp1 = np.random.randn(10); tmp2 = np.random.randn(19, 10); tmp3 = np.random.randn(13, 10); tmp4 = np.random.randn(14, 10)
        self.intmp = cpreconstruction(tmp1, [tmp2, tmp3, tmp4])

    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def testcp(self):
        l1, intmp1 = cpdecomposition(self.intmp, rank=10, init='svd', maxiters=100)
        f = cpreconstruction(l1, intmp1)
        check = np.linalg.norm(f - self.intmp)/np.linalg.norm(self.intmp)
        print(check)
        assert check <= 1e-3