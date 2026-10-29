"""Workspace-root launcher for the object-tracking application.

The implementation lives inside ``python/``; run ``python main.py`` here to
process the media in ``dataset/``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_ROOT = PROJECT_ROOT / "python"


def main() -> None:
    sys.path.insert(0, str(PYTHON_ROOT))
    runpy.run_path(str(PYTHON_ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()

