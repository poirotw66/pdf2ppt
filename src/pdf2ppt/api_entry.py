"""Console-script entry for ``pdf2ppt-api`` without importing FastAPI at module load.

Keeps ``pip install .`` (no ``[server]``) from exploding with a raw
``ModuleNotFoundError: uvicorn`` when the script is invoked; instead print a
short install hint and exit 3.
"""

from __future__ import annotations

import importlib
import sys


def run() -> None:
    try:
        api = importlib.import_module("pdf2ppt.api")
    except ImportError as error:
        print(
            "error: Review UI API dependencies are missing "
            f"({error}). Install with: pip install 'pdf2ppt[server]'",
            file=sys.stderr,
        )
        raise SystemExit(3) from error
    api.run()
