"""Pytest configuration: make the library and examples importable without install."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in ("src", "examples"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
