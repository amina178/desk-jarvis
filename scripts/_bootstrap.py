"""Make `deskjarvis` importable even without `pip install -e .`.

Editable installs rely on a .pth file in site-packages. Python 3.13
skips .pth files that have the macOS "hidden" flag, which iCloud-synced
folders (Desktop / Documents) can set - then `import deskjarvis` fails
although the package is installed. Putting src/ on sys.path directly
removes that dependency.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
