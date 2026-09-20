"""Progress tracker: run each module's suite against YOUR code.

Usage:
    .venv/bin/python progress.py            # KATA_IMPL=exercises (your code)
    KATA_IMPL=solutions .venv/bin/python progress.py   # reference health check

Exit code is nonzero while any module fails.
"""

import os
import re
import subprocess
import sys

MODULES = [
    ("01 Foundations", "tests/test_01_foundations.py"),
    ("02 Deep learning", "tests/test_02_deep_learning.py"),
    ("03 Training", "tests/test_03_training.py"),
    ("04 Architectures", "tests/test_04_architectures.py"),
    ("05 Optimization", "tests/test_05_optimization.py"),
    ("06 Regularization", "tests/test_06_regularization.py"),
    ("07 Sequences", "tests/test_07_sequences.py"),
    ("08 Generative", "tests/test_08_generative.py"),
    ("09 Capstone", "tests/test_09_capstone.py"),
    ("Harness", "tests/test_impl_selector.py"),
]

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_suite(path):
    env = dict(os.environ, KATA_IMPL=os.environ.get("KATA_IMPL", "exercises"))
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-q", "--tb=no"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    tail = (out.stdout + out.stderr).strip().splitlines()
    summary = tail[-1] if tail else ""
    mf = re.search(r"(\d+) failed", summary)
    mp = re.search(r"(\d+) passed", summary)
    if mf or mp:
        failed = int(mf.group(1)) if mf else 0
        return (failed == 0 and out.returncode == 0), summary
    return None, summary or "no summary"


def main():
    impl = os.environ.get("KATA_IMPL", "exercises")
    print(f"KATA_IMPL={impl}  ({'your code' if impl == 'exercises' else 'reference'})\n")
    print(f"{'module':<20}{'result'}")
    ok_all = True
    for name, path in MODULES:
        if not os.path.exists(os.path.join(ROOT, path)):
            print(f"{name:<20}missing")
            ok_all = False
            continue
        ok, summary = run_suite(path)
        mark = "PASS" if ok else ("FAIL" if ok is False else "??")
        print(f"{name:<20}{mark}  {summary}")
        ok_all = ok_all and bool(ok)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
