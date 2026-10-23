"""Batch folder conversion: discovery, isolation, resume, summaries, exit codes.

Uses digital fixture PDFs so no OCR / paddlepaddle is required (CI-safe).
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from pdf2ppt.batch import (
    EXIT_PARTIAL_FAILURE,
    EXIT_SUCCESS,
    EXIT_USAGE,
    discover_pdfs,
    format_summary_markdown,
    is_valid_pptx,
    plan_batch_outputs,
    run_batch,
)
from pdf2ppt.cli import build_batch_parser, build_parser, main
from pdf2ppt.core import ConversionOptions
from pdf2ppt.models import ConversionReport, PageResult, QualityScore

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _quality() -> QualityScore:
    return QualityScore(
        text_confidence=0.9,
        layout_overlap_score=0.9,
        style_recovery_score=0.9,
        editable_ratio=0.85,
    )


def _write_minimal_pptx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")


def _fake_ok_report(input_path: Path, output_path: Path) -> ConversionReport:
    page = PageResult(
        page_number=1,
        page_kind="digital",
        background_mode="elements",
        width_pt=612.0,
        height_pt=792.0,
        text_blocks=[],
        quality_score=_quality(),
        fallback_reason=None,
    )
    _write_minimal_pptx(output_path)
    return ConversionReport(
        input_path=str(input_path),
        output_path=str(output_path),
        pages=[page],
    )


@unittest.skipUnless((FIXTURES_DIR / "solid_background.pdf").exists(), "fixture missing")
class BatchDiscoveryTests(unittest.TestCase):
    def test_discover_non_recursive_sorted_skips_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.pdf").write_bytes(b"%PDF")
            (root / "a.PDF").write_bytes(b"%PDF")
            (root / ".secret.pdf").write_bytes(b"%PDF")
            (root / "note.txt").write_bytes(b"x")
            sub = root / "nested"
            sub.mkdir()
            (sub / "nested.pdf").write_bytes(b"%PDF")

            found = discover_pdfs(root, recursive=False)
            names = [path.name for path in found]
            self.assertEqual(names, ["a.PDF", "b.pdf"])

    def test_discover_recursive_preserves_order_and_skips_hidden_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "z.pdf").write_bytes(b"%PDF")
            nested = root / "sub"
            nested.mkdir()
            (nested / "m.pdf").write_bytes(b"%PDF")
            hidden = root / ".hidden"
            hidden.mkdir()
            (hidden / "nope.pdf").write_bytes(b"%PDF")

            found = discover_pdfs(root, recursive=True)
            keys = [path.relative_to(root).as_posix() for path in found]
            self.assertEqual(keys, ["sub/m.pdf", "z.pdf"])

    def test_collision_suffix_for_case_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a = root / "deck.pdf"
            b = root / "deck.PDF"
            a.write_bytes(b"%PDF-1")
            b.write_bytes(b"%PDF-2")
            out = root / "out"
            planned = plan_batch_outputs(
                [a, b],
                input_dir=root,
                output_dir=out,
                recursive=False,
            )
            outputs = [item.output_path.name for item in planned]
            self.assertEqual(outputs, ["deck.pptx", "deck__2.pptx"])

    def test_collision_plan_stable_under_shuffled_discovery_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lower = root / "a.pdf"
            upper = root / "a.PDF"
            lower.write_bytes(b"%PDF-lower")
            upper.write_bytes(b"%PDF-upper")
            out = root / "out"
            plans: set[tuple[str, ...]] = set()
            rng = random.Random(0)
            for _ in range(30):
                order = [lower, upper]
                rng.shuffle(order)
                # Mimic discover_pdfs sorting so plan input is always stable.
                ordered = sorted(
                    order,
                    key=lambda item: (
                        str(item.relative_to(root)).casefold(),
                        str(item.relative_to(root)),
                    ),
                )
                planned = plan_batch_outputs(
                    ordered,
                    input_dir=root,
                    output_dir=out,
                    recursive=False,
                )
                plans.add(tuple(f"{item.relative_key}->{item.output_path.name}" for item in planned))
            self.assertEqual(len(plans), 1)
            only = next(iter(plans))
            self.assertEqual(only, ("a.PDF->a.pptx", "a.pdf->a__2.pptx"))

    def test_recursive_skips_pdfs_inside_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "docs"
            nested.mkdir()
            (nested / "keep.pdf").write_bytes(b"%PDF")
            out = root / "out"
            out.mkdir()
            (out / "stale.pdf").write_bytes(b"%PDF")
            found = discover_pdfs(root, recursive=True, exclude_dirs=[out])
            keys = [path.relative_to(root).as_posix() for path in found]
            self.assertEqual(keys, ["docs/keep.pdf"])


@unittest.skipUnless((FIXTURES_DIR / "solid_background.pdf").exists(), "fixture missing")
class BatchRunTests(unittest.TestCase):
    def test_failure_isolation_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            good = input_dir / "good.pdf"
            bad = input_dir / "bad.pdf"
            shutil.copy(FIXTURES_DIR / "solid_background.pdf", good)
            bad.write_bytes(b"not-a-pdf")

            calls: list[str] = []

            def convert_fn(options: ConversionOptions, **kwargs: object) -> ConversionReport:
                calls.append(options.input_path.name)
                if options.input_path.name == "bad.pdf":
                    raise RuntimeError("corrupt pdf")
                return _fake_ok_report(options.input_path, options.output_path)

            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            summary, code = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                convert_fn=convert_fn,
                quiet=True,
            )
            self.assertEqual(code, EXIT_PARTIAL_FAILURE)
            self.assertEqual(sorted(calls), ["bad.pdf", "good.pdf"])
            statuses = {row.file: row.status for row in summary.results}
            self.assertEqual(statuses["good.pdf"], "ok")
            self.assertEqual(statuses["bad.pdf"], "failed")
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertTrue((output_dir / "summary.md").is_file())

    def test_fail_fast_stops_after_first_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            first = input_dir / "a.pdf"
            second = input_dir / "b.pdf"
            first.write_bytes(b"bad")
            shutil.copy(FIXTURES_DIR / "solid_background.pdf", second)
            seen: list[str] = []

            def convert_fn(options: ConversionOptions, **kwargs: object) -> ConversionReport:
                seen.append(options.input_path.name)
                if options.input_path.name == "a.pdf":
                    raise RuntimeError("boom")
                return _fake_ok_report(options.input_path, options.output_path)

            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            summary, code = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                convert_fn=convert_fn,
                fail_fast=True,
                quiet=True,
            )
            self.assertEqual(code, EXIT_PARTIAL_FAILURE)
            self.assertEqual(seen, ["a.pdf"])
            self.assertEqual(len(summary.results), 1)
            self.assertEqual(summary.results[0].status, "failed")

    def test_resume_and_skip_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            one = input_dir / "one.pdf"
            two = input_dir / "two.pdf"
            shutil.copy(FIXTURES_DIR / "solid_background.pdf", one)
            shutil.copy(FIXTURES_DIR / "solid_background.pdf", two)

            # Seed a prior successful summary for one.pdf and an existing pptx for two.pdf
            prior = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "results": [
                    {
                        "file": "one.pdf",
                        "output": str(output_dir / "one.pptx"),
                        "pages": 1,
                        "status": "ok",
                        "seconds": 0.1,
                        "ocr_used": False,
                        "ocr_pages": 0,
                        "raster_fallback_pages": 0,
                        "warnings_count": 0,
                        "planner_cost_usd": None,
                        "quality": {},
                        "error": None,
                        "ssim_mean": None,
                    }
                ],
                "totals": {},
            }
            (output_dir / "summary.json").write_text(json.dumps(prior), encoding="utf-8")
            for name in ("one.pptx", "two.pptx"):
                _write_minimal_pptx(output_dir / name)

            convert_calls: list[str] = []

            def convert_fn(options: ConversionOptions, **kwargs: object) -> ConversionReport:
                convert_calls.append(options.input_path.name)
                return _fake_ok_report(options.input_path, options.output_path)

            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            summary, code = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                convert_fn=convert_fn,
                resume=True,
                skip_existing=True,
                quiet=True,
            )
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(convert_calls, [])
            statuses = {row.file: row.status for row in summary.results}
            self.assertEqual(statuses["one.pdf"], "ok")  # earlier ok row (with metrics) is retained
            self.assertEqual(summary.results[0].pages, 1)
            self.assertEqual(statuses["two.pdf"], "skipped")  # no prior row to retain

            # Second resume (+ skip-existing) must keep the same ok count / pages.
            summary2, code2 = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                convert_fn=convert_fn,
                resume=True,
                skip_existing=True,
                quiet=True,
            )
            self.assertEqual(code2, EXIT_SUCCESS)
            self.assertEqual(convert_calls, [])
            self.assertEqual(summary2.totals["ok"], 1)
            self.assertEqual(summary2.totals["pages"], 1)

    def test_double_resume_retains_ok_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            for name in ("a.pdf", "b.pdf"):
                shutil.copy(FIXTURES_DIR / "solid_background.pdf", input_dir / name)

            convert_calls: list[str] = []

            def convert_fn(options: ConversionOptions, **kwargs: object) -> ConversionReport:
                convert_calls.append(options.input_path.name)
                return _fake_ok_report(options.input_path, options.output_path)

            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            first, code = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                convert_fn=convert_fn,
                quiet=True,
            )
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(first.totals["ok"], 2)
            self.assertEqual(first.totals["pages"], 2)
            self.assertEqual(sorted(convert_calls), ["a.pdf", "b.pdf"])
            convert_calls.clear()

            for _ in range(2):
                again, code = run_batch(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    options_template=template,
                    convert_fn=convert_fn,
                    resume=True,
                    quiet=True,
                )
                self.assertEqual(code, EXIT_SUCCESS)
                self.assertEqual(convert_calls, [])
                self.assertEqual(again.totals["ok"], 2)
                self.assertEqual(again.totals["pages"], 2)
                self.assertEqual(again.results[0].pages, 1)
                self.assertEqual(again.results[0].status, "ok")

    def test_resume_reconverts_when_pptx_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            pdf = input_dir / "gone.pdf"
            shutil.copy(FIXTURES_DIR / "solid_background.pdf", pdf)
            prior = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "results": [
                    {
                        "file": "gone.pdf",
                        "output": str(output_dir / "gone.pptx"),
                        "pages": 1,
                        "status": "ok",
                        "seconds": 0.2,
                        "ocr_used": False,
                        "ocr_pages": 0,
                        "raster_fallback_pages": 0,
                        "warnings_count": 0,
                        "planner_cost_usd": None,
                        "quality": {"score": 0.9},
                        "error": None,
                        "ssim_mean": None,
                    }
                ],
                "totals": {},
            }
            (output_dir / "summary.json").write_text(json.dumps(prior), encoding="utf-8")
            # pptx intentionally missing → resume must reconvert
            convert_calls: list[str] = []

            def convert_fn(options: ConversionOptions, **kwargs: object) -> ConversionReport:
                convert_calls.append(Path(options.output_path).name)
                return _fake_ok_report(options.input_path, options.output_path)

            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            summary, code = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                convert_fn=convert_fn,
                resume=True,
                quiet=True,
            )
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(convert_calls, ["gone.pptx.partial"])
            self.assertTrue(is_valid_pptx(output_dir / "gone.pptx"))
            self.assertEqual(summary.results[0].status, "ok")

    def test_skip_existing_rejects_truncated_pptx(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            output_dir.mkdir()
            shutil.copy(FIXTURES_DIR / "solid_background.pdf", input_dir / "a.pdf")
            (output_dir / "a.pptx").write_bytes(b"PK\x03\x04truncated-not-a-zip")
            convert_calls: list[str] = []

            def convert_fn(options: ConversionOptions, **kwargs: object) -> ConversionReport:
                convert_calls.append(options.input_path.name)
                return _fake_ok_report(options.input_path, options.output_path)

            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            summary, code = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                convert_fn=convert_fn,
                skip_existing=True,
                quiet=True,
            )
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(convert_calls, ["a.pdf"])
            self.assertEqual(summary.results[0].status, "ok")
            self.assertTrue(is_valid_pptx(output_dir / "a.pptx"))

    def test_empty_folder_warns_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            with self.assertLogs("pdf2ppt.batch", level="WARNING") as logs:
                summary, code = run_batch(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    options_template=template,
                    quiet=True,
                )
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(summary.totals["files"], 0)
            self.assertTrue(any("no PDF files found" in line for line in logs.output))
            self.assertTrue((output_dir / "summary.json").is_file())

    def test_missing_input_dir_does_not_create_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            template = ConversionOptions(
                input_path=root / "missing",
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            with self.assertRaises(Exception):
                run_batch(
                    input_dir=root / "missing",
                    output_dir=output_dir,
                    options_template=template,
                    quiet=True,
                )
            self.assertFalse(output_dir.exists())

    def test_summary_content_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            pdf = input_dir / "deck.pdf"
            shutil.copy(FIXTURES_DIR / "solid_background.pdf", pdf)

            def convert_fn(options: ConversionOptions, **kwargs: object) -> ConversionReport:
                report = _fake_ok_report(options.input_path, options.output_path)
                report.layout_planner = {"estimated_cost_usd": 0.0123, "calls": 1}
                return report

            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
            )
            summary, code = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                convert_fn=convert_fn,
                quiet=True,
            )
            self.assertEqual(code, EXIT_SUCCESS)
            row = summary.results[0]
            self.assertEqual(row.status, "ok")
            self.assertEqual(row.pages, 1)
            self.assertFalse(row.ocr_used)
            self.assertIn("score", row.quality)
            self.assertAlmostEqual(row.planner_cost_usd or 0.0, 0.0123)

            payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["totals"]["ok"], 1)
            with (output_dir / "summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["file"], "deck.pdf")
            self.assertEqual(rows[0]["status"], "ok")
            md = format_summary_markdown(summary)
            self.assertIn("| deck.pdf |", md)
            self.assertIn("**Totals:**", md)

    def test_real_fixture_conversion_smoke(self) -> None:
        """End-to-end on a digital fixture (no OCR) through the real pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "in"
            output_dir = root / "out"
            input_dir.mkdir()
            shutil.copy(FIXTURES_DIR / "solid_background.pdf", input_dir / "solid.pdf")
            shutil.copy(FIXTURES_DIR / "dark_background_light_text.pdf", input_dir / "dark.pdf")

            template = ConversionOptions(
                input_path=input_dir,
                output_path=output_dir / "x.pptx",
                report_path=output_dir / "x.report.json",
                ocr_model_root=root / "no-ocr-models",
            )
            summary, code = run_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                options_template=template,
                quiet=True,
            )
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(summary.totals["ok"], 2)
            self.assertTrue((output_dir / "solid.pptx").is_file())
            self.assertTrue((output_dir / "dark.pptx").is_file())
            self.assertGreater((output_dir / "solid.pptx").stat().st_size, 100)


class BatchCliTests(unittest.TestCase):
    def test_batch_parser_shares_conversion_flags(self) -> None:
        single = build_parser().parse_args(["a.pdf", "b.pptx", "--mode", "fast", "--ocr-batch-size", "5"])
        batch = build_batch_parser().parse_args(
            ["indir", "-o", "outdir", "--mode", "fast", "--ocr-batch-size", "5"]
        )
        self.assertEqual(single.mode, batch.mode)
        self.assertEqual(single.ocr_batch_size, batch.ocr_batch_size)

    def test_main_batch_missing_output_is_usage_error(self) -> None:
        code = main(["batch", "/tmp"])
        self.assertEqual(code, EXIT_USAGE)

    @patch("pdf2ppt.cli.run_batch")
    def test_main_batch_prints_markdown_and_returns_code(self, run_batch_mock: Mock) -> None:
        from pdf2ppt.batch import BatchFileResult, BatchSummary

        run_batch_mock.return_value = (
            BatchSummary(
                input_dir="/in",
                output_dir="/out",
                results=[
                    BatchFileResult(
                        file="a.pdf",
                        output="/out/a.pptx",
                        pages=1,
                        status="ok",
                        seconds=0.5,
                        ocr_used=False,
                        ocr_pages=0,
                        raster_fallback_pages=0,
                        warnings_count=0,
                        planner_cost_usd=None,
                        quality={"score": 0.9},
                    )
                ],
                totals={"files": 1, "ok": 1, "failed": 0, "skipped": 0, "pages": 1, "seconds": 0.5},
            ),
            EXIT_SUCCESS,
        )
        code = main(["batch", "indir", "-o", "outdir", "--quiet"])
        self.assertEqual(code, EXIT_SUCCESS)
        run_batch_mock.assert_called_once()

    def test_single_file_cli_unchanged_when_batch_not_used(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.pdf", "output.pptx"])
        self.assertEqual(args.input_pdf, Path("input.pdf"))
        self.assertEqual(args.output_pptx, Path("output.pptx"))
