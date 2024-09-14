from source.decompositions.cp import *
from source.decompositions.tt import *
from source.decompositions.cp_to_tt import cp_to_tt
import pytest


class Testcp:
    def setup_method(self, method):
        print(f"Setting up {method}")
        tmp1 = np.random.randn(10); tmp2 = np.random.randn(19, 10); tmp3 = np.random.randn(13, 10); tmp4 = np.random.randn(14, 10)
        tensor = cpreconstruction(tmp1, [tmp2, tmp3, tmp4])
        intcp = cpdecomposition(tensor, 10)

        cprecon = cpreconstruction(*intcp)
        inttt = cp_to_tt(*intcp)
        ttrecon = ttrec(inttt)
        self.check = np.linalg.norm(cprecon - ttrecon)


    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def testcp(self):
        print(self.check)
        assert self.check <= 1e-13