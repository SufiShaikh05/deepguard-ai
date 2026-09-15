from __future__ import annotations

import sys
from pathlib import Path

_ML_CORE_PATH = Path(__file__).resolve().parents[2] / "ml_core"
if str(_ML_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(_ML_CORE_PATH))
