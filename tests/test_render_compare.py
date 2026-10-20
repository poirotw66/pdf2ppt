"""Unit tests for slide-level fidelity / editability metrics.

LibreOffice-dependent paths are skipped when ``soffice`` is missing so the
default CI backend job stays green without the optional system package. A
separate report-only workflow job installs LibreOffice and exercises the
full render path.
"""

from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Emu, Inches, Pt

from pdf2ppt.render_compare import (
    CJK_FONT_FAMILY,
    EditabilityMetrics,
    FidelityMetrics,
    RenderCompareError,
    SlideQualityScore,
    build_pdf_text_mask,
    compare_page_images,
    compute_boundary_ring_mae,
    compute_edge_similarity,
    compute_masked_pixel_mae,
    compute_pixel_error,
    compute_ssim,
    convert_pptx_to_pdf,
    evaluate_pptx_against_pdf,
    extract_pptx_background_image,
    find_libreoffice_executable,
    inspect_pptx_editability,
    inspect_slide_editability,
    render_pdf_page_image,
    render_text_free_background,
    run_subprocess_with_process_group,
    summarize_slide_scores,
    text_token_f1,
    tokenize_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _solid(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color=color)


class SsimTests(unittest.TestCase):
    def test_identical_images_score_one(self) -> None:
        image = _solid((64, 48), (40, 80, 120))
        self.assertAlmostEqual(compute_ssim(image, image.copy()), 1.0, places=5)

    def test_very_different_images_score_lower(self) -> None:
        left = _solid((64, 48), (0, 0, 0))
        right = _solid((64, 48), (255, 255, 255))
        self.assertLess(compute_ssim(left, right), 0.3)

    def test_resize_mismatch_still_computes(self) -> None:
        reference = _solid((80, 60), (10, 20, 30))
        candidate = _solid((40, 30), (10, 20, 30))
        score = compute_ssim(reference, candidate)
        self.assertGreater(score, 0.95)


class PixelErrorTests(unittest.TestCase):
    def test_identical_zero_error(self) -> None:
        image = _solid((20, 20), (1, 2, 3))
        stats = compute_pixel_error(image, image.copy())
        self.assertEqual(stats["pixel_mae"], 0.0)
        self.assertEqual(stats["pixel_mse"], 0.0)

    def test_known_mae(self) -> None:
        left = _solid((2, 2), (0, 0, 0))
        right = _solid((2, 2), (10, 0, 0))
        stats = compute_pixel_error(left, right)
        # Mean abs over RGB: 10/3 per pixel
        self.assertAlmostEqual(stats["pixel_mae"], 10.0 / 3.0, places=4)

    def test_masked_mae_ignores_outside_mask(self) -> None:
        left = _solid((10, 10), (0, 0, 0))
        right = _solid((10, 10), (0, 0, 0))
        draw = ImageDraw.Draw(right)
        draw.rectangle((0, 0, 9, 9), fill=(30, 0, 0))
        draw.rectangle((2, 2, 5, 5), fill=(90, 0, 0))
        mask = Image.new("L", (10, 10), 0)
        ImageDraw.Draw(mask).rectangle((2, 2, 5, 5), fill=255)
        stats = compute_masked_pixel_mae(left, right, mask)
        self.assertAlmostEqual(float(stats["masked_mae"]), 30.0, places=4)
        ring = compute_boundary_ring_mae(left, right, mask, ring_px=1)
        self.assertGreater(ring, 0.0)


class InpaintRegionMetricTests(unittest.TestCase):
    def test_text_free_background_removes_native_text(self) -> None:
        pdf = FIXTURES_DIR / "solid_background.pdf"
        if not pdf.exists():
            self.skipTest("fixture missing")
        gt = render_text_free_background(pdf, 0, dpi=72)
        original = render_pdf_page_image(pdf, 0, dpi=72)
        mask = build_pdf_text_mask(pdf, 0, image_size=gt.size, padding_px=4)
        import numpy as np

        self.assertGreater(int(np.count_nonzero(np.array(mask) > 0)), 0)
        # Inside text boxes the redacted page should differ from the original.
        masked = compute_masked_pixel_mae(original, gt, mask)
        self.assertGreater(float(masked["masked_mae"]), 1.0)

    def test_extract_pptx_background_image(self) -> None:
        presentation = Presentation()
        presentation.slide_width = Inches(10)
        presentation.slide_height = Inches(7.5)
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color=(7, 8, 9)).save(buffer, format="PNG")
        buffer.seek(0)
        slide.shapes.add_picture(
            buffer,
            Emu(0),
            Emu(0),
            width=presentation.slide_width,
            height=presentation.slide_height,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bg.pptx"
            presentation.save(str(path))
            image = extract_pptx_background_image(path, slide_index=0)
        self.assertIsNotNone(image)
        assert image is not None
        self.assertEqual(image.getpixel((0, 0)), (7, 8, 9))


class EdgeSimilarityTests(unittest.TestCase):
    def test_matching_edges_high_f1(self) -> None:
        image = _solid((80, 80), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 60, 60), outline=(0, 0, 0), width=2)
        scores = compute_edge_similarity(image, image.copy())
        self.assertGreater(scores["edge_f1"], 0.9)

    def test_blank_vs_edged_low_f1(self) -> None:
        blank = _solid((80, 80), (255, 255, 255))
        edged = blank.copy()
        draw = ImageDraw.Draw(edged)
        draw.line((0, 0, 79, 79), fill=(0, 0, 0), width=3)
        scores = compute_edge_similarity(blank, edged)
        self.assertLess(scores["edge_f1"], 0.5)


class TextF1Tests(unittest.TestCase):
    def test_tokenize_cjk_and_latin(self) -> None:
        tokens = tokenize_text("Hello 世界, world!")
        self.assertIn("hello", tokens)
        self.assertIn("世界", tokens)
        self.assertIn("world", tokens)

    def test_identical_strings_f1_one(self) -> None:
        scores = text_token_f1("Alpha Beta Gamma", "alpha beta gamma")
        self.assertEqual(scores["text_f1"], 1.0)

    def test_partial_overlap(self) -> None:
        scores = text_token_f1("one two three", "two three four")
        self.assertAlmostEqual(scores["text_f1"], 2.0 / 3.0, places=5)

    def test_empty_both_is_perfect(self) -> None:
        scores = text_token_f1("", "")
        self.assertEqual(scores["text_f1"], 1.0)


class ComparePageImagesTests(unittest.TestCase):
    def test_fidelity_without_text(self) -> None:
        image = _solid((32, 32), (90, 90, 90))
        metrics = compare_page_images(image, image.copy())
        self.assertAlmostEqual(metrics.ssim, 1.0, places=5)
        self.assertIsNone(metrics.text_f1)
        self.assertIn("skipped", (metrics.text_f1_note or "").lower())

    def test_fidelity_with_text(self) -> None:
        image = _solid((32, 32), (90, 90, 90))
        metrics = compare_page_images(
            image,
            image.copy(),
            reference_text="hello world",
            candidate_text="hello world",
        )
        self.assertEqual(metrics.text_f1, 1.0)


class EditabilityTests(unittest.TestCase):
    def _blank_presentation(self) -> tuple[Presentation, Any]:
        presentation = Presentation()
        presentation.slide_width = Inches(10)
        presentation.slide_height = Inches(7.5)
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        return presentation, slide

    def test_textbox_only_slide_scores_above_zero(self) -> None:
        presentation, slide = self._blank_presentation()
        box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1))
        box.text_frame.text = "Editable title"
        metrics = inspect_slide_editability(
            slide,
            slide_width=int(presentation.slide_width),
            slide_height=int(presentation.slide_height),
        )
        self.assertEqual(metrics.text_box_count, 1)
        self.assertEqual(metrics.picture_count, 0)
        self.assertEqual(metrics.raster_background_area_ratio, 0.0)
        self.assertGreater(metrics.score, 0.15)

    def test_full_page_picture_raises_raster_ratio(self) -> None:
        presentation, slide = self._blank_presentation()
        # 1x1 PNG as a full-slide picture.
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color=(12, 34, 56)).save(buffer, format="PNG")
        buffer.seek(0)
        slide.shapes.add_picture(
            buffer,
            Emu(0),
            Emu(0),
            width=presentation.slide_width,
            height=presentation.slide_height,
        )
        metrics = inspect_slide_editability(
            slide,
            slide_width=int(presentation.slide_width),
            slide_height=int(presentation.slide_height),
        )
        self.assertEqual(metrics.picture_count, 1)
        self.assertGreaterEqual(metrics.raster_background_area_ratio, 0.85)
        # Picture-only full-bleed slide: low editability.
        self.assertLess(metrics.score, 0.2)

    def test_inspect_pptx_roundtrip(self) -> None:
        presentation, slide = self._blank_presentation()
        box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(2), Inches(0.5))
        box.text_frame.text = "hi"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pptx"
            presentation.save(str(path))
            metrics_list = inspect_pptx_editability(path)
        self.assertEqual(len(metrics_list), 1)
        self.assertEqual(metrics_list[0].text_box_count, 1)


class SummarizeTests(unittest.TestCase):
    def test_summary_means(self) -> None:
        fidelity = FidelityMetrics(ssim=0.8, pixel_mae=1.0, pixel_mse=2.0, edge_f1=0.5, text_f1=0.9)
        editability = EditabilityMetrics(
            text_box_count=2,
            picture_count=1,
            auto_shape_count=0,
            table_count=0,
            line_count=0,
            group_count=0,
            chart_count=0,
            other_shape_count=0,
            native_object_count=2,
            raster_background_area_ratio=1.0,
            score=0.4,
        )
        pages = [
            SlideQualityScore(page_number=1, fidelity=fidelity, editability=editability),
            SlideQualityScore(
                page_number=2,
                fidelity=FidelityMetrics(ssim=1.0, pixel_mae=0.0, pixel_mse=0.0, edge_f1=1.0, text_f1=None),
                editability=EditabilityMetrics(
                    text_box_count=0,
                    picture_count=0,
                    auto_shape_count=0,
                    table_count=0,
                    line_count=0,
                    group_count=0,
                    chart_count=0,
                    other_shape_count=0,
                    native_object_count=0,
                    raster_background_area_ratio=0.0,
                    score=0.2,
                ),
            ),
        ]
        summary = summarize_slide_scores(pages)
        self.assertEqual(summary["page_count"], 2)
        self.assertAlmostEqual(summary["fidelity"]["ssim_mean"], 0.9, places=5)
        self.assertAlmostEqual(summary["fidelity"]["text_f1_mean"], 0.9, places=5)
        self.assertAlmostEqual(summary["editability"]["score_mean"], 0.3, places=5)


@unittest.skipUnless(FIXTURES_DIR.joinpath("solid_background.pdf").exists(), "fixtures missing")
class FixtureRenderSmokeTests(unittest.TestCase):
    def test_render_pdf_page_image_from_fixture(self) -> None:
        image = render_pdf_page_image(FIXTURES_DIR / "solid_background.pdf", 0, dpi=72)
        self.assertGreater(image.width, 10)
        self.assertGreater(image.height, 10)
        self.assertEqual(image.mode, "RGB")


class ProcessGroupKillTests(unittest.TestCase):
    """Fake soffice that spawns a child and hangs — child must die on timeout."""

    def _write_hanging_fake_soffice(self, root: Path) -> tuple[Path, Path]:
        child_pid_file = root / "child.pid"
        ready_file = root / "ready"
        fake_soffice = root / "fake-soffice"
        fake_soffice.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"CHILD_PID_FILE={child_pid_file!s}\n"
            f"READY_FILE={ready_file!s}\n"
            "(\n"
            "  echo $$ > \"$CHILD_PID_FILE\"\n"
            "  # Ignore TERM so only a group kill (or KILL) reaps us.\n"
            "  trap '' TERM\n"
            "  while true; do sleep 30; done\n"
            ") &\n"
            "while [ ! -s \"$CHILD_PID_FILE\" ]; do sleep 0.01; done\n"
            "touch \"$READY_FILE\"\n"
            "while true; do sleep 30; done\n",
            encoding="utf-8",
        )
        fake_soffice.chmod(fake_soffice.stat().st_mode | stat.S_IXUSR)
        return fake_soffice, child_pid_file

    @staticmethod
    def _wait_for_pid_file(path: Path, *, timeout_s: float = 2.0) -> int:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if path.exists():
                raw = path.read_text(encoding="utf-8").strip()
                if raw.isdigit():
                    return int(raw)
            time.sleep(0.02)
        raise AssertionError(f"pid file not written: {path}")

    @staticmethod
    def _assert_pid_gone(pid: int, *, timeout_s: float = 2.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        raise AssertionError(f"orphan child pid={pid} still alive after timeout")

    @unittest.skipIf(sys.platform == "win32", "POSIX process-group kill tested here")
    def test_timeout_kills_child_process_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_soffice, child_pid_file = self._write_hanging_fake_soffice(root)

            with self.assertRaises(subprocess.TimeoutExpired):
                run_subprocess_with_process_group(
                    [str(fake_soffice)],
                    timeout_seconds=1.0,
                )

            child_pid = self._wait_for_pid_file(child_pid_file)
            self._assert_pid_gone(child_pid)

    @unittest.skipIf(sys.platform == "win32", "POSIX process-group kill tested here")
    def test_convert_pptx_to_pdf_timeout_kills_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_soffice, child_pid_file = self._write_hanging_fake_soffice(root)
            pptx_path = root / "deck.pptx"
            pptx_path.write_bytes(b"PK fake pptx")
            out_dir = root / "out"
            out_dir.mkdir()

            with self.assertRaises(RenderCompareError) as ctx:
                convert_pptx_to_pdf(
                    pptx_path,
                    out_dir,
                    timeout_seconds=1,
                    libreoffice=fake_soffice,
                )
            self.assertIn("timed out", str(ctx.exception).lower())

            child_pid = self._wait_for_pid_file(child_pid_file)
            self._assert_pid_gone(child_pid)


@unittest.skipUnless(find_libreoffice_executable() is not None, "LibreOffice not installed")
class LibreOfficeIntegrationTests(unittest.TestCase):
    def test_evaluate_minimal_pptx_against_solid_fixture(self) -> None:
        """End-to-end smoke: build a trivial PPTX and score it against a fixture PDF."""

        pdf_path = FIXTURES_DIR / "solid_background.pdf"
        self.assertTrue(pdf_path.exists())

        presentation = Presentation()
        # Match fixture page size (960 x 540 pt).
        presentation.slide_width = Pt(960)
        presentation.slide_height = Pt(540)
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        # Approximate the fixture: solid pale background + a title textbox.
        buffer = io.BytesIO()
        Image.new("RGB", (96, 54), color=(245, 248, 252)).save(buffer, format="PNG")
        buffer.seek(0)
        slide.shapes.add_picture(
            buffer,
            Emu(0),
            Emu(0),
            width=presentation.slide_width,
            height=presentation.slide_height,
        )
        box = slide.shapes.add_textbox(Pt(72), Pt(72), Pt(400), Pt(40))
        box.text_frame.text = "Solid Background"

        with tempfile.TemporaryDirectory() as tmp:
            pptx_path = Path(tmp) / "approx.pptx"
            presentation.save(str(pptx_path))
            report = evaluate_pptx_against_pdf(
                pptx_path,
                pdf_path,
                dpi=72,
                ocr_engine=None,
                work_dir=Path(tmp) / "render",
            )

        self.assertEqual(len(report.pages), 1)
        self.assertIn("libreoffice", report.renderer)
        self.assertGreater(report.pages[0].fidelity.ssim, 0.3)
        self.assertEqual(report.pages[0].editability.text_box_count, 1)
        self.assertEqual(CJK_FONT_FAMILY, "Noto Sans CJK TC")


if __name__ == "__main__":
    unittest.main()
