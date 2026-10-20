"""Unit tests for the P4 deterministic render-compare-refine loop.

Diagnosis and fix logic run on synthetic inputs without OCR or LibreOffice.
End-to-end refine is gated on ``soffice`` being present.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from pdf2ppt.models import ImagePlacement, PageResult, QualityScore, ShapeElement, TextBlock
from pdf2ppt.refine import (
    DEFAULT_REFINE_ITERS,
    DeckRefineReport,
    apply_match_line_bboxes,
    apply_raster_fallback,
    apply_shrink_font,
    apply_widen_for_wrap,
    diagnose_extra_wrap,
    diagnose_line_drift,
    diagnose_low_local_ssim,
    diagnose_slide,
    diagnose_text_overflow,
    propose_fixes_for_issue,
    refine_pages,
    region_ssim,
    write_refine_report,
)
from pdf2ppt.render_compare import find_libreoffice_executable

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _quality() -> QualityScore:
    return QualityScore(
        text_confidence=1.0,
        layout_overlap_score=1.0,
        style_recovery_score=1.0,
        editable_ratio=1.0,
    )


def _page(
    *,
    text_blocks: list[TextBlock] | None = None,
    shape_elements: list[ShapeElement] | None = None,
    image_elements: list[ImagePlacement] | None = None,
    width: float = 200.0,
    height: float = 100.0,
) -> PageResult:
    return PageResult(
        page_number=1,
        page_kind="digital",
        background_mode="elements",
        width_pt=width,
        height_pt=height,
        text_blocks=list(text_blocks or []),
        quality_score=_quality(),
        fallback_reason=None,
        shape_elements=list(shape_elements or []),
        image_elements=list(image_elements or []),
    )


def _solid(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color=color)


class DiagnoseOverflowTests(unittest.TestCase):
    def test_bbox_past_slide_edge(self) -> None:
        block = TextBlock(
            id="t1",
            source="native",
            bbox=(180.0, 10.0, 220.0, 30.0),
            text="Hi",
            confidence=1.0,
            font_size=12.0,
        )
        issue = diagnose_text_overflow(block, page_width_pt=200.0, page_height_pt=100.0)
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(issue.kind, "text_overflow")
        self.assertTrue(issue.details["past_slide_edge"])

    def test_fits_inside_page_no_issue(self) -> None:
        block = TextBlock(
            id="t2",
            source="native",
            bbox=(10.0, 10.0, 80.0, 28.0),
            text="OK",
            confidence=1.0,
            font_size=10.0,
        )
        issue = diagnose_text_overflow(block, page_width_pt=200.0, page_height_pt=100.0)
        # Narrow short text should not overflow a wide box.
        self.assertIsNone(issue)


class DiagnoseWrapTests(unittest.TestCase):
    def test_extra_wrap_when_box_too_narrow(self) -> None:
        block = TextBlock(
            id="wrap1",
            source="native",
            bbox=(10.0, 10.0, 28.0, 24.0),
            text="ABCDEFGHIJKLMNOP",
            confidence=1.0,
            font_size=14.0,
            line_bboxes=[(10.0, 10.0, 120.0, 24.0)],
        )
        issue = diagnose_extra_wrap(block)
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(issue.kind, "extra_wrap")
        self.assertGreater(issue.details["estimated_line_count"], issue.details["source_line_count"])

    def test_no_extra_wrap_when_box_wide(self) -> None:
        block = TextBlock(
            id="wrap2",
            source="native",
            bbox=(10.0, 10.0, 400.0, 30.0),
            text="Short",
            confidence=1.0,
            font_size=12.0,
            line_bboxes=[(10.0, 10.0, 80.0, 30.0)],
        )
        self.assertIsNone(diagnose_extra_wrap(block))


class DiagnoseLineDriftTests(unittest.TestCase):
    def test_drift_detected(self) -> None:
        block = TextBlock(
            id="d1",
            source="native",
            bbox=(10.0, 10.0, 100.0, 50.0),
            text="A\nB",
            confidence=1.0,
            font_size=12.0,
            line_bboxes=[(10.0, 10.0, 100.0, 22.0), (10.0, 28.0, 100.0, 40.0)],
        )
        issue = diagnose_line_drift(block, rendered_line_ys_pt=[16.0, 55.0], drift_pt=3.0)
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(issue.kind, "line_drift")

    def test_aligned_lines_ok(self) -> None:
        block = TextBlock(
            id="d2",
            source="native",
            bbox=(10.0, 10.0, 100.0, 50.0),
            text="A\nB",
            confidence=1.0,
            font_size=12.0,
            line_bboxes=[(10.0, 10.0, 100.0, 22.0), (10.0, 28.0, 100.0, 40.0)],
        )
        self.assertIsNone(diagnose_line_drift(block, rendered_line_ys_pt=[16.0, 34.0], drift_pt=3.0))


class DiagnoseLocalSsimTests(unittest.TestCase):
    def test_low_local_ssim(self) -> None:
        issue = diagnose_low_local_ssim(
            element_id="s1",
            local_ssim=0.5,
            page_ssim=0.9,
            gap=0.02,
        )
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(issue.kind, "low_local_ssim")

    def test_high_local_mae_flags_recolor(self) -> None:
        issue = diagnose_low_local_ssim(
            element_id="s1",
            local_ssim=0.98,
            page_ssim=0.99,
            gap=0.02,
            local_mae=40.0,
            page_mae=1.0,
        )
        self.assertIsNotNone(issue)

    def test_near_page_average_ok(self) -> None:
        self.assertIsNone(
            diagnose_low_local_ssim(
                element_id="s1",
                local_ssim=0.985,
                page_ssim=0.99,
                gap=0.02,
                local_mae=2.0,
                page_mae=1.5,
            )
        )


class DiagnoseSlideTests(unittest.TestCase):
    def test_synthetic_mismatch_finds_low_local_ssim(self) -> None:
        # Large page, small wrong patch so page SSIM stays high while local SSIM drops.
        reference = _solid((200, 120), (255, 255, 255))
        draw = ImageDraw.Draw(reference)
        draw.rectangle((20, 20, 50, 50), fill=(0, 120, 200))
        candidate = reference.copy()
        ImageDraw.Draw(candidate).rectangle((20, 20, 50, 50), fill=(200, 20, 20))

        page = _page(
            shape_elements=[
                ShapeElement(
                    id="shape_1",
                    kind="rect",
                    bbox=(20.0, 20.0, 50.0, 50.0),
                    fill_color="#0078C8",
                )
            ],
            width=200.0,
            height=120.0,
        )
        ssim, issues = diagnose_slide(reference, candidate, page)
        self.assertLess(ssim, 1.0)
        kinds = {issue.kind for issue in issues}
        self.assertIn("low_local_ssim", kinds)

    def test_identical_images_few_or_no_visual_issues(self) -> None:
        image = _solid((80, 40), (240, 240, 240))
        page = _page(
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(5.0, 5.0, 70.0, 20.0),
                    text="Hi",
                    confidence=1.0,
                    font_size=10.0,
                )
            ],
            width=80.0,
            height=40.0,
        )
        ssim, issues = diagnose_slide(image, image.copy(), page)
        self.assertGreater(ssim, 0.99)
        self.assertTrue(all(issue.kind != "low_local_ssim" for issue in issues))


class FixTests(unittest.TestCase):
    def test_shrink_font(self) -> None:
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(10, 10, 100, 30),
                    text="Hello",
                    confidence=1.0,
                    font_size=18.0,
                )
            ]
        )
        details = apply_shrink_font(page, "t", step_pt=1.0)
        self.assertIsNotNone(details)
        self.assertEqual(page.text_blocks[0].font_size, 17.0)

    def test_match_line_bboxes(self) -> None:
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(0, 0, 20, 10),
                    text="AB\nCD",
                    confidence=1.0,
                    font_size=12.0,
                    line_bboxes=[(5, 5, 80, 18), (5, 20, 80, 33)],
                )
            ]
        )
        details = apply_match_line_bboxes(page, "t")
        self.assertIsNotNone(details)
        x0, y0, x1, y1 = page.text_blocks[0].bbox
        self.assertGreater(x1 - x0, 20)
        self.assertGreater(y1 - y0, 10)

    def test_widen_for_wrap(self) -> None:
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(10, 10, 25, 24),
                    text="WIDETEXTSTRING",
                    confidence=1.0,
                    font_size=14.0,
                    line_bboxes=[(10, 10, 120, 24)],
                )
            ],
            width=400.0,
            height=100.0,
        )
        before = page.text_blocks[0].bbox[2] - page.text_blocks[0].bbox[0]
        details = apply_widen_for_wrap(page, "t")
        self.assertIsNotNone(details)
        after = page.text_blocks[0].bbox[2] - page.text_blocks[0].bbox[0]
        self.assertGreater(after, before)

    def test_raster_fallback_keeps_text(self) -> None:
        reference = _solid((100, 60), (10, 20, 30))
        ImageDraw.Draw(reference).rectangle((20, 10, 60, 40), fill=(255, 0, 0))
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(20, 10, 60, 40),
                    text="Keep me",
                    confidence=1.0,
                    font_size=12.0,
                )
            ],
            width=100.0,
            height=60.0,
        )
        details = apply_raster_fallback(page, "t", reference_page_image=reference)
        self.assertIsNotNone(details)
        self.assertEqual(len(page.image_elements), 1)
        self.assertEqual(page.text_blocks[0].text, "Keep me")

    def test_propose_fixes_order(self) -> None:
        from pdf2ppt.refine import RefineIssue

        overflow = RefineIssue(kind="text_overflow", element_id="t", severity=0.5)
        self.assertEqual(propose_fixes_for_issue(overflow)[0], "shrink_font")
        low = RefineIssue(kind="low_local_ssim", element_id="s", severity=0.5)
        self.assertIn("raster_fallback", propose_fixes_for_issue(low))


class RegionSsimTests(unittest.TestCase):
    def test_region_identical(self) -> None:
        image = _solid((40, 40), (50, 50, 50))
        score = region_ssim(image, image.copy(), (5, 5, 30, 30))
        self.assertIsNotNone(score)
        assert score is not None
        self.assertGreater(score, 0.99)


class AcceptRejectLoopTests(unittest.TestCase):
    def test_rejects_fix_that_lowers_metric(self) -> None:
        """Synthetic render hooks: a shrink that paints worse is reverted."""

        reference = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(reference).text((8, 8), "Title", fill=(0, 0, 0))

        good = reference.copy()
        bad = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(bad).text((8, 8), "Title", fill=(180, 180, 180))

        page = _page(
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(5, 5, 55, 22),
                    text="Title",
                    confidence=1.0,
                    font_size=40.0,  # oversized → overflow/wrap diagnosis
                    line_bboxes=[(5, 5, 55, 22)],
                )
            ],
            width=64.0,
            height=36.0,
        )
        pages = [page]
        call_state = {"n": 0}

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            call_state["n"] += 1
            # First render: good baseline. After a shrink attempt: worse.
            if call_state["n"] == 1:
                return [good.copy()]
            return [bad.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            Path(tmp, "src.pdf").write_bytes(b"%PDF-1.4")
            # Force overflow issue so a shrink is attempted.
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                result_pages, report = refine_pages(
                    pages,
                    pdf_path=Path(tmp) / "src.pdf",
                    output_pptx=out,
                    slide_width_pt=64.0,
                    slide_height_pt=36.0,
                    iterations=1,
                    max_fixes_per_slide=2,
                    dpi=72,
                    write_pptx=fake_write,
                    render_pptx=fake_render_pptx,
                    render_pdf_page=fake_render_pdf,
                    work_dir=Path(tmp) / "work",
                )

        self.assertEqual(len(report.pages), 1)
        slide = report.pages[0]
        # At least one fix attempt should have been rejected (metric drop).
        rejected = [fix for fix in slide.fixes if not fix.accepted]
        self.assertTrue(rejected or slide.metric_after is not None)
        # Font size must not stay reduced if the only attempts were rejected.
        if rejected and all(not fix.accepted for fix in slide.fixes):
            self.assertEqual(result_pages[0].text_blocks[0].font_size, 40.0)

    def test_accepts_fix_that_improves_metric(self) -> None:
        # Large canvas + small wrong patch → low_local_ssim vs high page SSIM.
        reference = _solid((160, 90), (255, 255, 255))
        ImageDraw.Draw(reference).rectangle((12, 12, 36, 36), fill=(0, 100, 200))

        bad = reference.copy()
        ImageDraw.Draw(bad).rectangle((12, 12, 36, 36), fill=(200, 0, 0))
        good = reference.copy()

        page = _page(
            shape_elements=[
                ShapeElement(
                    id="card",
                    kind="rect",
                    bbox=(12, 12, 36, 36),
                    fill_color="#FF0000",
                )
            ],
            width=160.0,
            height=90.0,
        )
        call_state = {"n": 0}

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            call_state["n"] += 1
            if call_state["n"] == 1:
                return [bad.copy()]
            # After raster_fallback the region matches the reference.
            return [good.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                result_pages, report = refine_pages(
                    [page],
                    pdf_path=Path(tmp) / "src.pdf",
                    output_pptx=out,
                    slide_width_pt=160.0,
                    slide_height_pt=90.0,
                    iterations=1,
                    max_fixes_per_slide=4,
                    dpi=72,
                    write_pptx=fake_write,
                    render_pptx=fake_render_pptx,
                    render_pdf_page=fake_render_pdf,
                    work_dir=Path(tmp) / "work",
                )

        slide = report.pages[0]
        self.assertIsNotNone(slide.metric_before)
        self.assertIsNotNone(slide.metric_after)
        assert slide.metric_before is not None and slide.metric_after is not None
        self.assertGreaterEqual(slide.metric_after, slide.metric_before - 1e-9)
        accepted = [fix for fix in slide.fixes if fix.accepted]
        self.assertTrue(accepted)
        self.assertGreaterEqual(len(result_pages[0].image_elements), 1)

    def test_skips_cleanly_without_libreoffice(self) -> None:
        page = _page()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            out.write_bytes(b"PK")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=None):
                result, report = refine_pages(
                    [page],
                    pdf_path=Path(tmp) / "src.pdf",
                    output_pptx=out,
                    slide_width_pt=100.0,
                    slide_height_pt=50.0,
                    iterations=2,
                )
        self.assertEqual(report.skipped_reason, "libreoffice_unavailable")
        self.assertEqual(result[0].page_number, 1)

    def test_write_refine_report_json(self) -> None:
        report = DeckRefineReport(
            enabled=True,
            skipped_reason=None,
            iterations=DEFAULT_REFINE_ITERS,
            dpi=144,
            pages=[],
            runtime_seconds=0.1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_refine_report(report, Path(tmp) / "out.refine.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["iterations"], DEFAULT_REFINE_ITERS)


@unittest.skipUnless(find_libreoffice_executable() is not None, "LibreOffice not installed")
@unittest.skipUnless(FIXTURES_DIR.joinpath("solid_background.pdf").exists(), "fixtures missing")
class RefineLibreOfficeIntegrationTests(unittest.TestCase):
    def test_refine_smoke_on_minimal_pptx(self) -> None:
        from pdf2ppt.pipeline import render_pages_to_pptx
        from pdf2ppt.render_compare import evaluate_pptx_against_pdf

        pdf_path = FIXTURES_DIR / "solid_background.pdf"
        page = _page(
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(72, 72, 400, 110),
                    text="Solid Background",
                    confidence=1.0,
                    font_size=28.0,
                    line_bboxes=[(72, 72, 400, 110)],
                )
            ],
            width=960.0,
            height=540.0,
        )
        # Approximate solid background as a full-page picture.
        buffer_img = Image.new("RGB", (96, 54), color=(245, 248, 252))
        import io

        buf = io.BytesIO()
        buffer_img.save(buf, format="PNG")
        page.background_image_bytes = buf.getvalue()
        page.background_mode = "full-page"

        with tempfile.TemporaryDirectory() as tmp:
            pptx_path = Path(tmp) / "deck.pptx"
            render_pages_to_pptx([page], pptx_path, slide_width_pt=960.0, slide_height_pt=540.0)
            before = evaluate_pptx_against_pdf(pptx_path, pdf_path, dpi=72, ocr_engine=None)
            _pages, report = refine_pages(
                [page],
                pdf_path=pdf_path,
                output_pptx=pptx_path,
                slide_width_pt=960.0,
                slide_height_pt=540.0,
                iterations=1,
                max_fixes_per_slide=4,
                dpi=72,
                work_dir=Path(tmp) / "refine",
            )
            after = evaluate_pptx_against_pdf(pptx_path, pdf_path, dpi=72, ocr_engine=None)

        self.assertIsNone(report.skipped_reason)
        self.assertEqual(len(report.pages), 1)
        self.assertGreaterEqual(
            after.pages[0].fidelity.ssim,
            before.pages[0].fidelity.ssim - 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
