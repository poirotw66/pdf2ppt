"""Messaging for optional extras ([ocr] / [server]) and CORS env parsing."""

from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pdf2ppt import api_entry
from pdf2ppt.api import _cors_allow_origins
from pdf2ppt.cli import main
from pdf2ppt.core import OcrInitializationError
from pdf2ppt.ocr import OcrEngine
from pdf2ppt.paths import get_repo_root, resolve_repo_relative_path


class OcrExtraMessagingTests(unittest.TestCase):
    def test_ocr_import_error_mentions_ocr_extra(self) -> None:
        engine = OcrEngine(
            "ch",
            model_root=None,
            use_doc_orientation=False,
            use_textline_orientation=False,
            use_doc_unwarping=False,
            det_thresh=None,
            det_box_thresh=None,
            drop_score=None,
        )
        # ``sys.modules[name] = None`` makes ``import name`` raise ImportError.
        with patch.dict(sys.modules, {"paddleocr": None}):
            with self.assertRaises(OcrInitializationError) as ctx:
                engine._get_engine()
        message = str(ctx.exception)
        self.assertIn("pdf2ppt[ocr]", message)
        self.assertIn("paddlepaddle==3.2.0", message)

    def test_cli_ocr_initialization_error_exits_3(self) -> None:
        stderr = io.StringIO()
        with (
            patch("pdf2ppt.cli.convert_pdf", side_effect=OcrInitializationError("install pdf2ppt[ocr]")),
            patch("sys.stderr", stderr),
        ):
            code = main(["input.pdf", "output.pptx"])
        self.assertEqual(code, 3)
        self.assertIn("error: install pdf2ppt[ocr]", stderr.getvalue())


class ServerExtraMessagingTests(unittest.TestCase):
    def test_api_entry_missing_server_extra_exits_3(self) -> None:
        stderr = io.StringIO()
        real_import_module = api_entry.importlib.import_module

        def fake_import_module(name: str, package: str | None = None):
            if name == "pdf2ppt.api":
                raise ImportError("No module named 'uvicorn'")
            return real_import_module(name, package)

        with (
            patch.object(api_entry.importlib, "import_module", side_effect=fake_import_module),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as ctx,
        ):
            api_entry.run()
        self.assertEqual(ctx.exception.code, 3)
        text = stderr.getvalue()
        self.assertIn("error:", text)
        self.assertIn("pdf2ppt[server]", text)


class CorsOriginsTests(unittest.TestCase):
    def test_blank_and_comma_only_fall_back_to_defaults(self) -> None:
        with patch.dict(os.environ, {"PDF2PPT_CORS_ORIGINS": ""}, clear=False):
            self.assertEqual(
                _cors_allow_origins(),
                ["http://localhost:5173", "http://127.0.0.1:5173"],
            )
        with patch.dict(os.environ, {"PDF2PPT_CORS_ORIGINS": " , , "}, clear=False):
            self.assertEqual(
                _cors_allow_origins(),
                ["http://localhost:5173", "http://127.0.0.1:5173"],
            )

    def test_star_origin_rejected(self) -> None:
        with patch.dict(os.environ, {"PDF2PPT_CORS_ORIGINS": "*"}, clear=False):
            self.assertEqual(
                _cors_allow_origins(),
                ["http://localhost:5173", "http://127.0.0.1:5173"],
            )

    def test_explicit_origins_honored(self) -> None:
        with patch.dict(
            os.environ,
            {"PDF2PPT_CORS_ORIGINS": "http://localhost:8080, http://127.0.0.1:8080"},
            clear=False,
        ):
            self.assertEqual(
                _cors_allow_origins(),
                ["http://localhost:8080", "http://127.0.0.1:8080"],
            )


class Pdf2PptRootTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("PDF2PPT_ROOT", None)
        get_repo_root.cache_clear()

    def test_pdf2ppt_root_redirects_relative_model_dir(self) -> None:
        get_repo_root.cache_clear()
        with patch.dict(os.environ, {"PDF2PPT_ROOT": "/data"}, clear=False):
            get_repo_root.cache_clear()
            self.assertEqual(get_repo_root(), Path("/data").resolve())
            self.assertEqual(resolve_repo_relative_path("model"), Path("/data/model").resolve())


if __name__ == "__main__":
    unittest.main()
