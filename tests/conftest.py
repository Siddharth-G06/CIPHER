# tests/conftest.py
# ------------------
# Adds the project root to sys.path so that 'from src.xxx import yyy'
# works correctly when pytest is run from any working directory.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
