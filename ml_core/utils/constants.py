"""
Single source of truth for label conventions. Every other file that
needs to know "0 = real, 1 = fake" imports it from here - never
redefine this list anywhere else.
"""

CLASS_NAMES: list[str] = ["real", "fake"]
REAL_LABEL: int = 0
FAKE_LABEL: int = 1
