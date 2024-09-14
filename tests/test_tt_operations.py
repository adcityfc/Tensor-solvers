from source.decompositions.tt import *
from source.decompositions.tt_operations import * 
import pytest


class Testttprop:
    def setup_method(self, method):
        print(f"Setting up {method}")
        # Create a tensor
        X = np.random.randn(7, 5, 5)
        Y = np.random.randn(7, 5, 5)
        rank = [1, 7, 5, 1]

        # Perform Tucker train decomposition
        self.factorsx = ttsvd(X, rank=rank)
        self.factorsy = ttsvd(Y, rank=rank)

        # Operations
        # Addition
        self.Zadd = X + Y

        # Multiply
        self.Zmult = X * Y

        # Create a tensor
        tmp1 = np.random.rand(1, 5, 5); tmp2 = np.random.rand(5, 9, 8); tmp3 = np.random.rand(8, 7, 1)
        self.factors = [tmp1, tmp2, tmp3]

        # test matvec
        matA = np.random.randn(27, 64)
        vecx = np.random.randn(64)
        self.res = matA@vecx
        self.Att = mat_to_tt(matA, [3]*3, [4]*3, [1, 100, 100, 1])
        self.xtt = vec_to_tt(vecx, [4]*3, [1, 100, 100, 1])

    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def test_add(self):
        factorsz = tt_add(self.factorsx, self.factorsy)
        Zrec = ttrec(factorsz)
        check = np.linalg.norm(self.Zadd - Zrec)/np.linalg.norm(self.Zadd)
        assert check < 1e-14, f"Reconstruction error too high and = {check}"

    def test_multiply(self):
        factorsz = tt_multiply(self.factorsx, self.factorsy)
        Zrec = ttrec(factorsz)
        check = np.linalg.norm(self.Zmult - Zrec)/np.linalg.norm(self.Zmult)
        assert check < 1e-14, f"Reconstruction error too high and = {check}"

    def test_dot(self):
        A = ttrec(self.factors)
        check = tt_dot(self.factors, self.factors) - np.sum(A*A)
        assert check < 1e-14, f"Reconstruction error too high and = {check}"

    def test_matvec(self):
        Axtt = tt_matvec(self.Att, self.xtt)
        restt = ttrec(Axtt)
        check = np.linalg.norm(self.res - restt.reshape((-1,), order='F'))/np.linalg.norm(self.res)
        assert check < 1e-14, f"Reconstruction error too high and = {check}"
