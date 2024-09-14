from tests.test_utils import test_diff_opt
import pytest


class Testdiv:
    def setup_method(self, method):
        print(f"Setting up {method}")
        self.errors = test_diff_opt(n=100, d=2, meth='cd', option=1)

    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def testdiv1(self):
        assert self.errors[0] <= 1e-12

    def testdiv2(self):
        assert self.errors[-1] <= 1e-1

class Testlap:
    def setup_method(self, method):
        print(f"Setting up {method}")
        self.errors = test_diff_opt(n=100, d=2, meth='cd', option=2)

    def teardown_method(self, method):
        print(f"Tearing down {method}")

    def testlap1(self):
        assert self.errors[0] <= 1e-12

    def testlap2(self):
        assert self.errors[-1] <= 1e-1