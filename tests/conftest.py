"""Tests import `deskjarvis` from src/ and helper scripts from scripts/,
whether or not the package was pip-installed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
