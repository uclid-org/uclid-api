from uclid.builder import Uadd

__author__ = "Federico Mora"
__copyright__ = "Federico Mora"
__license__ = "MIT"


def test_uadd():
    assert Uadd([1, 2]).__inject__() == "(1) + (2)"
