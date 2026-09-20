"""Implementation selector: tests evaluate LEARNER code by default.

KATA_IMPL=exercises (default) -> import from exercises/ (your code; red→green).
KATA_IMPL=solutions            -> import from solutions/ (reference health check).

`ref()` always loads solutions/ — for tests that need a known-good helper
(e.g. wrapping the reference SGD inside Lookahead).
"""

import os
from importlib import import_module

IMPL = os.environ.get("KATA_IMPL", "exercises")
assert IMPL in ("exercises", "solutions"), f"bad KATA_IMPL={IMPL!r}"


def load(modname: str):
    """Load exercises|solutions.<modname> per KATA_IMPL (e.g. '01_foundations.losses')."""
    return import_module(f"{IMPL}.{modname}")


def ref(modname: str):
    """Always load solutions.<modname>."""
    return import_module(f"solutions.{modname}")
