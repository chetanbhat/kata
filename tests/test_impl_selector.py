"""Harness self-test: the KATA_IMPL selector resolves the right packages.

Mode-independent: asserts the loader contract, not any implementation.
"""

from tests import impl


def test_impl_is_a_known_package():
    assert impl.IMPL in ("exercises", "solutions")


def test_load_follows_kata_impl():
    mod = impl.load("01_foundations.losses")
    assert mod.__name__.split(".")[0] == impl.IMPL
    assert hasattr(mod, "cross_entropy")


def test_ref_always_loads_solutions():
    mod = impl.ref("01_foundations.losses")
    assert mod.__name__.split(".")[0] == "solutions"
