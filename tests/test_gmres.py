import numpy as np
# from source.decompositions.tt import *
from source.solvers import gmres, tt_gmres
import pytest


class Testgmres:
    def setup_method(self, method):
        print(f"Setting up {method}")
        A = np.random.randn(4, 4); b = np.random.randn(4)
        tsoln = np.linalg.inv(A)@b
        x, res = gmres(A, b, np.random.randn(4), maxiter=4)
        self.check = np.linalg.norm(tsoln - x)/np.linalg.norm(tsoln)


    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def test_tt(self):
        print(self.check)
        assert self.check <= 1e-14


class Test_ttgmres:
    def setup_method(self, method):
        from source.decompositions.tt_operations import mat_to_tt, vec_to_tt
        print(f"Setting up {method}")
        # TT-gmres test
        A = np.random.randn(8, 8); b = np.random.randn(8)
        self.tsoln = np.linalg.inv(A)@b

        self.Att = mat_to_tt(A, [2]*3, [2]*3, [1, 1000, 1000, 1])
        self.btt = vec_to_tt(b, [2]*3, rank = [1, 1000, 1000, 1])
        self.x0tt = [np.random.randn(*(s.shape)) for s in self.btt]


    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def test_tt2(self):
        from source.decompositions.tt_operations import tt_to_vecmat
        x, res = tt_gmres(self.Att, self.btt, self.x0tt, maxiter=8, rank=[1, 1000, 1000, 1], eps=1e-14)
        check = np.linalg.norm(self.tsoln - tt_to_vecmat(x))
        print(check)
        assert check <= 1e-12