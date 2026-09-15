"""Ensures backend/ and ml_core/ are on sys.path for pytest."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
_ML_CORE_DIR = _BACKEND_DIR.parent / "ml_core"

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

if str(_ML_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_CORE_DIR))
