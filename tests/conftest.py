"""Pytest configuration: make the kata root importable so tests can load
``solutions`` / ``exercises`` packages (including digit-prefixed dirs)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
