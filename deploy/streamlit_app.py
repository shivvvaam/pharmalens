"""Streamlit Community Cloud entrypoint.

Community Cloud installs dependencies from the entrypoint's folder before the repo
root, so this file lets the slim deploy/requirements.txt win over the root uv.lock
(which also contains the evaluation stack). It simply runs the real app.py.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
