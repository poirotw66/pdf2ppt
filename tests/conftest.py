"""Shared pytest fixtures and markers.

CI installs ``paddleocr`` without ``paddlepaddle`` (see ``.github/workflows/ci.yml``).
Mark tests that need a real Paddle runtime with ``@pytest.mark.requires_ocr`` (or the
unittest helper ``skip_unless_paddlepaddle``).
"""

from __future__ import annotations

import importlib.util
import unittest

import pytest


def paddlepaddle_available() -> bool:
    return importlib.util.find_spec("paddle") is not None


requires_ocr = pytest.mark.skipif(
    not paddlepaddle_available(),
    reason="paddlepaddle not installed (CI installs paddleocr only; use mocks or [ocr] extra)",
)


def skip_unless_paddlepaddle() -> None:
    """Call at the start of a unittest method that needs a real PaddleOCR stack."""
    if not paddlepaddle_available():
        raise unittest.SkipTest(
            "paddlepaddle not installed (CI installs paddleocr only; use mocks or [ocr] extra)"
        )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_ocr: needs paddlepaddle / real PaddleOCR (skipped in CI without [ocr])",
    )
