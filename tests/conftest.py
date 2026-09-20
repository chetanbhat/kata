"""Pytest configuration: make the kata root importable so tests can load
``solutions`` / ``exercises`` packages (including digit-prefixed dirs)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Matplotlib's default config dir may be read-only; point it at /tmp before
# any test module imports pyplot (capstone plotting).
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kata-mpl-config")
