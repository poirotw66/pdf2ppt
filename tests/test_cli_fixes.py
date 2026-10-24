"""Focused regression tests for small CLI/library consistency fixes."""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pdf2ppt import core as core_mod
from pdf2ppt import timeouts as timeouts_mod
from pdf2ppt.cli import (
    EXIT_INTERRUPTED,
    build_parser,
    main,
)
from pdf2ppt.core import (
    DEFAULT_INPAINT_ENGINE,
    DEFAULT_INPAINT_LAMA_PATCH_HYBRID,
    ConversionOptions,
)
from pdf2ppt.inpainting_engines import uses_lama_patch_hybrid


class SubcommandHelpTests(unittest.TestCase):
    def _assert_help_ok(self, argv: list[str], *must_contain: str) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            code = main(argv)
        self.assertEqual(code, 0, msg=f"argv={argv!r} stderr={stderr.getvalue()!r}")
        text = stdout.getvalue() + stderr.getvalue()
        for needle in must_contain:
            self.assertIn(needle, text, msg=f"missing {needle!r} in help for {argv!r}")

    def test_emit_ir_help_exits_0_and_mentions_flags(self) -> None:
        self._assert_help_ok(
            ["emit-ir", "--help"],
            "pdf2ppt emit-ir",
            "--inpaint-engine",
            "--mode",
        )
        self._assert_help_ok(["emit-ir", "-h"], "pdf2ppt emit-ir", "--report")

    def test_from_ir_help_exits_0_and_mentions_flags(self) -> None:
        self._assert_help_ok(
            ["from-ir", "--help"],
            "pdf2ppt from-ir",
            "--inpaint-engine",
            "output_pptx",
        )
        self._assert_help_ok(["from-ir", "-h"], "pdf2ppt from-ir", "--report")

    def test_batch_and_root_help_still_exit_0(self) -> None:
        self._assert_help_ok(["batch", "--help"], "pdf2ppt batch", "--jobs", "-o")
        self._assert_help_ok(["--help"], "--emit-ir", "--from-ir", "--mode")
        self._assert_help_ok(["-h"], "--emit-ir")

    def test_module_entry_emit_ir_help_exits_0(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "pdf2ppt", "emit-ir", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        combined = completed.stdout + completed.stderr
        self.assertIn("pdf2ppt emit-ir", combined)
        self.assertIn("--inpaint-engine", combined)


class InpaintHybridDefaultTests(unittest.TestCase):
    def test_dataclass_default_matches_cli_parser_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.pdf", "output.pptx"])
        self.assertEqual(args.inpaint_engine, DEFAULT_INPAINT_ENGINE)
        cli_hybrid = uses_lama_patch_hybrid(args.inpaint_engine)
        field_default = next(
            f.default for f in fields(ConversionOptions) if f.name == "inpaint_lama_patch_hybrid"
        )
        self.assertEqual(field_default, cli_hybrid)
        self.assertEqual(field_default, DEFAULT_INPAINT_LAMA_PATCH_HYBRID)
        self.assertFalse(DEFAULT_INPAINT_LAMA_PATCH_HYBRID)


class PageTimeoutConstantTests(unittest.TestCase):
    def test_core_and_timeouts_share_same_default_object(self) -> None:
        self.assertIs(
            core_mod.DEFAULT_PAGE_TIMEOUT_SECONDS,
            timeouts_mod.DEFAULT_PAGE_TIMEOUT_SECONDS,
        )
        self.assertEqual(core_mod.DEFAULT_PAGE_TIMEOUT_SECONDS, 120.0)


class PymupdfImportTests(unittest.TestCase):
    def test_cli_import_raises_no_fitz_deprecation(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::DeprecationWarning",
                "-c",
                "import pdf2ppt.cli",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        combined = (completed.stdout + completed.stderr).lower()
        self.assertNotIn("fitz", combined)
        self.assertNotIn("deprecated", combined)

    def test_help_emits_no_fitz_warning(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "pdf2ppt", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        combined = (completed.stdout + completed.stderr).lower()
        self.assertNotIn("fitz", combined)


class SingleFileKeyboardInterruptTests(unittest.TestCase):
    def test_keyboard_interrupt_exits_130_without_traceback_or_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            partial = out.with_name(out.name + ".partial")

            def fake_convert(options, **_kwargs):  # noqa: ANN001
                options.output_path.write_bytes(b"truncated-pptx")
                raise KeyboardInterrupt()

            stderr = io.StringIO()
            stdout = io.StringIO()
            with (
                patch("pdf2ppt.cli.convert_pdf", side_effect=fake_convert),
                patch("sys.stderr", stderr),
                patch("sys.stdout", stdout),
            ):
                code = main(["input.pdf", str(out)])

            self.assertEqual(code, EXIT_INTERRUPTED)
            err = stderr.getvalue()
            self.assertIn("interrupted:", err)
            self.assertNotIn("Traceback", err)
            self.assertNotIn("KeyboardInterrupt", err)
            self.assertFalse(out.exists())
            self.assertFalse(partial.exists())

    def test_successful_convert_still_works_with_atomic_helper(self) -> None:
        with patch(
            "pdf2ppt.cli.convert_pdf",
            return_value=SimpleNamespace(
                input_path="input.pdf",
                output_path="output.pptx",
                pages=[object()],
            ),
        ):
            code = main(["input.pdf", "output.pptx"])
        self.assertEqual(code, 0)


class AtomicPartialSidecarTests(unittest.TestCase):
    def test_refine_report_default_uses_final_pptx_stem(self) -> None:
        captured: list[object] = []

        def fake_convert(options, **_kwargs):  # noqa: ANN001
            captured.append(options)
            return SimpleNamespace(
                input_path="in.pdf",
                output_path=str(options.output_path),
                pages=[object()],
            )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            with patch("pdf2ppt.cli.convert_pdf", side_effect=fake_convert):
                code = main(["in.pdf", str(out), "--refine"])
            self.assertEqual(code, 0)
            self.assertEqual(len(captured), 1)
            options = captured[0]
            self.assertEqual(options.output_path, out.with_name(out.name + ".partial"))
            self.assertEqual(options.refine_report_path, out.with_suffix(".refine.json"))
            self.assertEqual(options.report_path, out.with_suffix(".report.json"))
            self.assertFalse(str(options.refine_report_path).endswith(".pptx.refine.json"))

    def test_explicit_refine_report_path_is_preserved(self) -> None:
        captured: list[object] = []

        def fake_convert(options, **_kwargs):  # noqa: ANN001
            captured.append(options)
            return SimpleNamespace(
                input_path="in.pdf",
                output_path=str(options.output_path),
                pages=[object()],
            )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            custom = Path(tmp) / "custom-refine.json"
            with patch("pdf2ppt.cli.convert_pdf", side_effect=fake_convert):
                code = main(["in.pdf", str(out), "--refine", "--refine-report", str(custom)])
            self.assertEqual(code, 0)
            self.assertEqual(captured[0].refine_report_path, custom)


if __name__ == "__main__":
    unittest.main()
