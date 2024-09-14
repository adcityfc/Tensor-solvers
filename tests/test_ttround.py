from source.decompositions.tt import *
from source.decompositions.tt_operations import tt_add
import pytest


class Testtt:
    def setup_method(self, method):
        print(f"Setting up {method}")
        # Create a tensor
        tmprank = [1, 2, 3, 1]
        tmp1 = np.random.randn(tmprank[0], 8, tmprank[1]); tmp2 = np.random.randn(tmprank[1], 10, tmprank[2]); tmp3 = np.random.randn(tmprank[2], 8, tmprank[3])
        self.f1 = [tmp1, tmp2, tmp3]
        self.rank = tmprank
        


    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def test_tt(self):
        f2 = tt_add(self.f1, self.f1)
        f1rec = ttrec(self.f1)
        f2rec = ttrec(f2)
        
        f2round = tt_rounding(f2, rank=self.rank); f2roundrec = ttrec(f2round)

        check = np.linalg.norm(f2roundrec - 2*f1rec)/np.linalg.norm(f1rec)
        print("Discrepancy between non-rounded and original tensor", np.linalg.norm(f2rec - 2*f1rec)/np.linalg.norm(f1rec))
        print("Discrepancy between rounded and original tensor", check)
        
        assert check <= 1e-14