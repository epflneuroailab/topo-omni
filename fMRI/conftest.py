"""Make the repo root importable so `import core` works when running `pytest` in
place, before anyone runs `pip install -e core/`. (importlib import-mode does not
manipulate sys.path, so we do it here.) Once core is pip-installed per env this is
harmless — the installed package still resolves first.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
