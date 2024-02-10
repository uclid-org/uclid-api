__author__ = "Federico Mora"
__copyright__ = "Federico Mora"
__license__ = "MIT"


def test_empty_module():
    from uclid import Module

    m = Module("empty")
    # assert str(m).split() == "module empty { }".split()
    assert "module empty {" in str(m)
