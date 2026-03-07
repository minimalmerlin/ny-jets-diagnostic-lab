"""Import anchor for the src-based project package."""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "jets_project"
__path__ = [str(_SRC_PACKAGE)]
