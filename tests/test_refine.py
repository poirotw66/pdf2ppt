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

from pdf2ppt.cli import build_parser
from pdf2ppt.models import (
    ImagePlacement,
    PageResult,
    QualityScore,
    ShapeElement,
    TextBlock,
    TextParagraph,
)
from pdf2ppt.refine import (
    DEFAULT_MAX_ATTEMPTS_PER_ELEMENT,
    DEFAULT_MAX_ATTEMPTS_PER_SLIDE,
    DEFAULT_MIN_FONT_RATIO,
    DEFAULT_MIN_SOFFICE_TIMEOUT_SECONDS,
    DEFAULT_REFINE_ITERS,
    DEFAULT_SHRINK_MIN_FONT_SIZE_PT,
    METRIC_IMPROVE_EPSILON,
    DeckRefineReport,
    RefineIssue,
    apply_match_line_bboxes,
    apply_raster_fallback,
    apply_shrink_font,
    apply_widen_for_wrap,
    diagnose_extra_wrap,
    diagnose_line_drift,
    diagnose_low_local_ssim,
    diagnose_slide,
    diagnose_text_overflow,
    min_soffice_timeout_seconds,
    order_pages_for_refine,
    page_issue_severity,
    page_refine_severity,
    page_time_allowance_seconds,
    pptx_content_digest,
    propose_fixes_for_issue,
    refine_pages,
    region_ssim,
    select_baseline_page_indices,
    shrink_font_floor,
    write_refine_report,
)
from pdf2ppt.render_compare import RenderCompareError, find_libreoffice_executable

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
        self.assertTrue(issue.details["render_evidence"])

    def test_measurement_only_overflow_is_demoted(self) -> None:
        # Wide measured text in a narrow box, but no rendered candidate → no issue
        # without render evidence (measurement alone does not emit when candidate is None
        # and box is on-page... actually measurement_overflow alone still emits demoted).
        block = TextBlock(
            id="t2",
            source="native",
            bbox=(10.0, 10.0, 28.0, 24.0),
            text="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            confidence=1.0,
            font_size=18.0,
        )
        issue = diagnose_text_overflow(block, page_width_pt=200.0, page_height_pt=100.0)
        if issue is not None:
            self.assertFalse(issue.details["render_evidence"])
            self.assertLessEqual(issue.severity, 0.2)


class DiagnoseWrapTests(unittest.TestCase):
    def test_extra_wrap_requires_render_evidence(self) -> None:
        block = TextBlock(
            id="wrap1",
            source="native",
            bbox=(10.0, 10.0, 28.0, 24.0),
            text="ABCDEFGHIJKLMNOP",
            confidence=1.0,
            font_size=14.0,
            line_bboxes=[(10.0, 10.0, 120.0, 24.0)],
        )
        # Without a candidate image, measurement-only wrap is not emitted.
        self.assertIsNone(diagnose_extra_wrap(block))

    def test_extra_wrap_from_ink_rows(self) -> None:
        # Two ink bands inside a box whose source claims one line.
        image = _solid((80, 50), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 50, 16), fill=(0, 0, 0))
        draw.rectangle((12, 28, 50, 32), fill=(0, 0, 0))
        block = TextBlock(
            id="wrap2",
            source="native",
            bbox=(10.0, 10.0, 55.0, 40.0),
            text="OneLine",
            confidence=1.0,
            font_size=12.0,
            line_bboxes=[(10.0, 10.0, 55.0, 20.0)],
        )
        issue = diagnose_extra_wrap(block, candidate=image, page_width_pt=80.0, page_height_pt=50.0)
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertTrue(issue.details["render_evidence"])
        self.assertGreater(issue.details["estimated_line_count"], issue.details["source_line_count"])


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
            element_kind="shape",
            local_ssim=0.5,
            page_ssim=0.9,
            gap=0.02,
        )
        self.assertIsNotNone(issue)
        assert issue is not None
        self.assertEqual(issue.kind, "low_local_ssim")
        self.assertTrue(issue.details["render_evidence"])

    def test_high_local_mae_flags_recolor(self) -> None:
        issue = diagnose_low_local_ssim(
            element_id="s1",
            element_kind="shape",
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
                element_kind="shape",
                local_ssim=0.985,
                page_ssim=0.99,
                gap=0.02,
                local_mae=2.0,
                page_mae=1.5,
            )
        )


class DiagnoseSlideTests(unittest.TestCase):
    def test_synthetic_mismatch_finds_low_local_ssim(self) -> None:
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
        # Render-evidence issues sort first.
        self.assertTrue(issues[0].has_render_evidence)

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
    def test_shrink_font_scales_paragraphs(self) -> None:
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(10, 10, 100, 30),
                    text="Hello\nWorld",
                    confidence=1.0,
                    font_size=18.0,
                    paragraphs=[
                        TextParagraph(text="Hello", font_size=18.0),
                        TextParagraph(text="World", font_size=16.0),
                    ],
                )
            ]
        )
        details = apply_shrink_font(page, "t", step_pt=1.0, min_font_ratio=0.5)
        self.assertIsNotNone(details)
        assert details is not None
        self.assertTrue(details["paragraph_sizes_scaled"])
        self.assertEqual(page.text_blocks[0].font_size, 17.0)
        paras = page.text_blocks[0].paragraphs or []
        self.assertEqual(paras[0].font_size, 17.0)
        self.assertEqual(paras[1].font_size, 15.0)

    def test_shrink_font_respects_ratio_and_absolute_floor(self) -> None:
        # 14pt with 95% ratio → floor 13.3; absolute 8 → floor 13.3 (blocks 14→13).
        self.assertAlmostEqual(shrink_font_floor(14.0), max(8.0, 14.0 * 0.95), places=5)
        self.assertEqual(DEFAULT_MIN_FONT_RATIO, 0.95)
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(10, 10, 100, 30),
                    text="Title",
                    confidence=1.0,
                    font_size=14.0,
                    paragraphs=[TextParagraph(text="Title", font_size=14.0)],
                )
            ]
        )
        # 14 → 13.5 is allowed; a further 0.5pt step clamps to the 13.3 floor;
        # another attempt is a no-op.
        details = apply_shrink_font(
            page,
            "t",
            step_pt=0.5,
            min_font_size_pt=DEFAULT_SHRINK_MIN_FONT_SIZE_PT,
            min_font_ratio=DEFAULT_MIN_FONT_RATIO,
            source_font_size=14.0,
            source_paragraph_sizes=[14.0],
        )
        self.assertIsNotNone(details)
        self.assertAlmostEqual(page.text_blocks[0].font_size or 0.0, 13.5, places=5)
        floor = shrink_font_floor(14.0)
        clamped = apply_shrink_font(
            page,
            "t",
            step_pt=0.5,
            min_font_size_pt=DEFAULT_SHRINK_MIN_FONT_SIZE_PT,
            min_font_ratio=DEFAULT_MIN_FONT_RATIO,
            source_font_size=14.0,
            source_paragraph_sizes=[14.0],
        )
        self.assertIsNotNone(clamped)
        self.assertAlmostEqual(page.text_blocks[0].font_size or 0.0, floor, places=5)
        self.assertIsNone(
            apply_shrink_font(
                page,
                "t",
                step_pt=0.5,
                min_font_size_pt=DEFAULT_SHRINK_MIN_FONT_SIZE_PT,
                min_font_ratio=DEFAULT_MIN_FONT_RATIO,
                source_font_size=14.0,
                source_paragraph_sizes=[14.0],
            )
        )

    def test_shrink_font_absolute_floor_for_small_source(self) -> None:
        # 8pt * 0.95 = 7.6, but absolute floor is 8 → cannot shrink.
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(10, 10, 100, 30),
                    text="Tiny",
                    confidence=1.0,
                    font_size=8.0,
                )
            ]
        )
        self.assertIsNone(
            apply_shrink_font(
                page,
                "t",
                step_pt=0.5,
                min_font_size_pt=8.0,
                min_font_ratio=0.95,
                source_font_size=8.0,
            )
        )
        self.assertEqual(page.text_blocks[0].font_size, 8.0)

    def test_widen_bbox_pins_font_size(self) -> None:
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(10, 10, 25, 24),
                    text="WIDETEXTSTRING",
                    confidence=1.0,
                    font_size=18.0,
                    line_bboxes=[(10, 10, 120, 24)],
                )
            ],
            width=400.0,
            height=100.0,
        )
        details = apply_widen_for_wrap(page, "t")
        self.assertIsNotNone(details)
        assert details is not None
        self.assertTrue(details["pin_font_size"])
        self.assertTrue(page.text_blocks[0].pin_font_size)
        self.assertEqual(page.text_blocks[0].font_size, 18.0)

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

    def test_raster_fallback_never_on_text(self) -> None:
        reference = _solid((100, 60), (10, 20, 30))
        page = _page(
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(20, 10, 60, 40),
                    text="Title",
                    confidence=1.0,
                    font_size=12.0,
                )
            ],
            width=100.0,
            height=60.0,
        )
        self.assertIsNone(apply_raster_fallback(page, "title", reference_page_image=reference))
        self.assertEqual(len(page.image_elements), 0)

    def test_raster_fallback_erases_overlapping_text_glyphs(self) -> None:
        reference = _solid((100, 60), (200, 200, 200))
        ImageDraw.Draw(reference).rectangle((10, 10, 80, 50), fill=(0, 100, 200))
        # White "glyphs" overlapping the shape region.
        ImageDraw.Draw(reference).rectangle((20, 15, 70, 28), fill=(255, 255, 255))
        page = _page(
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(20, 15, 70, 28),
                    text="Title",
                    confidence=1.0,
                    font_size=12.0,
                )
            ],
            shape_elements=[
                ShapeElement(
                    id="card",
                    kind="rect",
                    bbox=(10, 10, 80, 50),
                    fill_color="#0064C8",
                )
            ],
            width=100.0,
            height=60.0,
        )
        details = apply_raster_fallback(page, "card", reference_page_image=reference)
        self.assertIsNotNone(details)
        assert details is not None
        self.assertTrue(details["text_glyphs_erased"])
        self.assertEqual(len(page.image_elements), 1)
        # Title text block remains editable.
        self.assertEqual(page.text_blocks[0].text, "Title")
        # Crop should not keep the bright glyph band untouched: mean in that
        # local region should move toward the fill/median, not stay pure white.
        from io import BytesIO

        crop = Image.open(BytesIO(page.image_elements[0].png_bytes)).convert("RGB")
        # Title band relative to shape crop origin (10,10).
        import numpy as np

        band = np.array(crop.crop((10, 5, 60, 18)))
        mean_luma = float(band.mean())
        self.assertLess(mean_luma, 250.0)

    def test_propose_fixes_never_rasters_text(self) -> None:
        from pdf2ppt.refine import RefineIssue

        overflow = RefineIssue(
            kind="text_overflow",
            element_id="t",
            severity=0.5,
            details={"element_kind": "text", "render_evidence": True},
        )
        self.assertNotIn("raster_fallback", propose_fixes_for_issue(overflow))
        shape_issue = RefineIssue(
            kind="low_local_ssim",
            element_id="s",
            severity=0.8,
            details={"element_kind": "shape", "render_evidence": True},
        )
        self.assertEqual(propose_fixes_for_issue(shape_issue), ["raster_fallback"])


class RegionSsimTests(unittest.TestCase):
    def test_region_identical(self) -> None:
        image = _solid((40, 40), (50, 50, 50))
        score = region_ssim(image, image.copy(), (5, 5, 30, 30))
        self.assertIsNotNone(score)
        assert score is not None
        self.assertGreater(score, 0.99)


class AcceptRejectLoopTests(unittest.TestCase):
    def test_rejects_neutral_metric(self) -> None:
        """Equal SSIM must not be accepted under the strict-improvement rule."""

        reference = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(reference).text((8, 8), "Title", fill=(0, 0, 0))
        same = reference.copy()

        page = _page(
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(5, 5, 55, 22),
                    text="Title",
                    confidence=1.0,
                    font_size=40.0,
                    line_bboxes=[(5, 5, 55, 22)],
                )
            ],
            width=64.0,
            height=36.0,
        )
        call_state = {"n": 0}

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            call_state["n"] += 1
            return [same.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch("pdf2ppt.refine.pptx_content_digest", side_effect=["aaa", "bbb", "bbb", "ccc"]):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.5):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                            _pages, report = refine_pages(
                                [page],
                                pdf_path=Path(tmp) / "src.pdf",
                                output_pptx=out,
                                slide_width_pt=64.0,
                                slide_height_pt=36.0,
                                iterations=2,
                                max_fixes_per_slide=4,
                                dpi=72,
                                write_pptx=fake_write,
                                render_pptx=fake_render_pptx,
                                render_pdf_page=fake_render_pdf,
                                work_dir=Path(tmp) / "work",
                            )

        slide = report.pages[0]
        accepted = [fix for fix in slide.fixes if fix.accepted]
        self.assertEqual(accepted, [])
        # No improving iteration consumed by neutral outcomes.
        self.assertEqual(slide.iterations, 0)

    def test_accepts_only_strict_ssim_improvement(self) -> None:
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
            return [good.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        digests = iter(["base", "trial", "done"])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch(
                    "pdf2ppt.refine.pptx_content_digest",
                    side_effect=lambda _p: next(digests, "done"),
                ):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.4):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
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
        assert slide.metric_before is not None and slide.metric_after is not None
        self.assertGreater(slide.metric_after, slide.metric_before + METRIC_IMPROVE_EPSILON - 1e-9)
        accepted = [fix for fix in slide.fixes if fix.accepted]
        self.assertTrue(accepted)
        self.assertEqual(accepted[0].action, "raster_fallback")
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

    def test_render_failure_skips_and_restores(self) -> None:
        page = _page(
            text_blocks=[
                TextBlock(
                    id="t",
                    source="native",
                    bbox=(10, 10, 80, 30),
                    text="Hello",
                    confidence=1.0,
                    font_size=14.0,
                )
            ]
        )
        written_paths: list[Path] = []

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"PK fake")
            written_paths.append(path)
            return path

        def failing_render(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            raise RenderCompareError("soffice exited 1")

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return _solid((64, 36), (255, 255, 255))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                result, report = refine_pages(
                    [page],
                    pdf_path=Path(tmp) / "src.pdf",
                    output_pptx=out,
                    slide_width_pt=100.0,
                    slide_height_pt=50.0,
                    iterations=2,
                    write_pptx=fake_write,
                    render_pptx=failing_render,
                    render_pdf_page=fake_render_pdf,
                    work_dir=Path(tmp) / "work",
                )

            self.assertEqual(report.skipped_reason, "render_failed")
            # Original font size preserved in returned pages.
            self.assertEqual(result[0].text_blocks[0].font_size, 14.0)
            # Last accepted PPTX was restored to disk (at least one write to output path).
            self.assertTrue(any(path.resolve() == out.resolve() for path in written_paths))
            self.assertTrue(out.exists())

    def test_mid_loop_render_failure_sets_deck_skipped_reason(self) -> None:
        """Baseline succeeds; a trial soffice failure must set deck skipped_reason."""

        reference = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(reference).text((8, 8), "Title", fill=(0, 0, 0))
        baseline = reference.copy()

        page = _page(
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(5, 5, 55, 22),
                    text="Title",
                    confidence=1.0,
                    font_size=40.0,
                    line_bboxes=[(5, 5, 55, 22)],
                )
            ],
            width=64.0,
            height=36.0,
        )
        call_state = {"n": 0}

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def flaky_render(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            call_state["n"] += 1
            if call_state["n"] == 1:
                return [baseline.copy()]
            raise RenderCompareError("soffice exited mid-loop")

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        digests = iter(["base", "trial1", "trial2", "done"])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch(
                    "pdf2ppt.refine.pptx_content_digest",
                    side_effect=lambda _p: next(digests, "done"),
                ):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.5):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                            _pages, report = refine_pages(
                                [page],
                                pdf_path=Path(tmp) / "src.pdf",
                                output_pptx=out,
                                slide_width_pt=64.0,
                                slide_height_pt=36.0,
                                iterations=2,
                                max_fixes_per_slide=4,
                                dpi=72,
                                write_pptx=fake_write,
                                render_pptx=flaky_render,
                                render_pdf_page=fake_render_pdf,
                                work_dir=Path(tmp) / "work",
                            )

        self.assertEqual(report.skipped_reason, "render_failed")
        self.assertEqual(report.pages[0].skipped_reason, "render_failed")

    def test_time_budget_spreads_across_pages(self) -> None:
        """With a shared budget, later pages still get a chance (fair share)."""

        pages = [
            _page(
                text_blocks=[
                    TextBlock(
                        id=f"t{index}",
                        source="native",
                        bbox=(5, 5, 40, 20),
                        text=f"P{index}",
                        confidence=1.0,
                        font_size=12.0,
                    )
                ],
                width=64.0,
                height=36.0,
            )
            for index in range(4)
        ]
        for index, page in enumerate(pages):
            page.page_number = index + 1

        render_timeouts: list[int] = []

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            render_timeouts.append(int(timeout_seconds))
            # One baseline image per page in the deck.
            return [_solid((64, 36), (255, 255, 255)) for _ in pages]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return _solid((64, 36), (255, 255, 255))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch("pdf2ppt.refine.pptx_content_digest", return_value="digest"):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.5):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                            _pages, report = refine_pages(
                                pages,
                                pdf_path=Path(tmp) / "src.pdf",
                                output_pptx=out,
                                slide_width_pt=64.0,
                                slide_height_pt=36.0,
                                iterations=1,
                                max_fixes_per_slide=1,
                                max_attempts_per_slide=1,
                                time_budget_seconds=60.0,
                                dpi=72,
                                write_pptx=fake_write,
                                render_pptx=fake_render_pptx,
                                render_pdf_page=fake_render_pdf,
                                work_dir=Path(tmp) / "work",
                            )

        self.assertEqual(len(report.pages), 4)
        self.assertTrue(all(page.skipped_reason != "time_budget" for page in report.pages))
        self.assertNotIn("render_failed", [page.skipped_reason for page in report.pages])
        # Deck baseline is a single soffice call.
        self.assertEqual(report.soffice_call_count, 1)
        self.assertTrue(render_timeouts)
        floor = int(min_soffice_timeout_seconds(None))
        self.assertTrue(all(floor <= value <= 180 for value in render_timeouts))

    def test_baseline_is_one_deck_render(self) -> None:
        """N-page decks must not issue N baseline soffice calls."""

        page_count = 12
        pages = [
            _page(
                text_blocks=[
                    TextBlock(
                        id=f"t{index}",
                        source="native",
                        bbox=(5, 5, 40, 20),
                        text=f"P{index}",
                        confidence=1.0,
                        font_size=12.0,
                    )
                ],
                width=64.0,
                height=36.0,
            )
            for index in range(page_count)
        ]
        for index, page in enumerate(pages):
            page.page_number = index + 1

        render_calls: list[int] = []

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            render_calls.append(len(pages))
            return [_solid((64, 36), (255, 255, 255)) for _ in pages]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return _solid((64, 36), (255, 255, 255))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch("pdf2ppt.refine.pptx_content_digest", return_value="digest"):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.5):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                            _pages, report = refine_pages(
                                pages,
                                pdf_path=Path(tmp) / "src.pdf",
                                output_pptx=out,
                                slide_width_pt=64.0,
                                slide_height_pt=36.0,
                                iterations=1,
                                max_fixes_per_slide=1,
                                max_attempts_per_slide=1,
                                time_budget_seconds=30.0,
                                dpi=72,
                                write_pptx=fake_write,
                                render_pptx=fake_render_pptx,
                                render_pdf_page=fake_render_pdf,
                                work_dir=Path(tmp) / "work",
                            )

        # Clean pages: only the shared baseline soffice call.
        self.assertEqual(len(render_calls), 1)
        self.assertEqual(report.soffice_call_count, 1)
        self.assertEqual(len(report.pages), page_count)
        self.assertTrue(all(page.skipped_reason != "time_budget" for page in report.pages))

    def test_page_order_prefers_severity_then_early_index(self) -> None:
        """Short budget must refine high-severity / early pages, not only the tail."""

        page_count = 8
        pages = [
            _page(
                shape_elements=[
                    ShapeElement(
                        id=f"card_{index}",
                        kind="rect",
                        bbox=(12, 12, 40, 40),
                        fill_color="#FF0000",
                    )
                ],
                width=64.0,
                height=36.0,
            )
            for index in range(page_count)
        ]
        for index, page in enumerate(pages):
            page.page_number = index + 1

        # Page 0 and 1 look wrong (red vs blue); later pages match (no issues).
        # Severity on early pages should win over later clean pages.
        reference_good = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(reference_good).rectangle((12, 12, 40, 40), fill=(0, 100, 200))
        reference_bad_target = reference_good.copy()
        candidate_bad = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(candidate_bad).rectangle((12, 12, 40, 40), fill=(200, 0, 0))

        trial_pages: list[int] = []
        digests = iter([f"d{i}" for i in range(40)])

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            label = str(pptx_path)
            if "baseline" in label or pptx_path.name.endswith("out.pptx") or "deck" in str(pptx_path):
                # Shared baseline: early pages mismatched, late pages matched.
                images = []
                for index in range(page_count):
                    if index <= 1:
                        images.append(candidate_bad.copy())
                    else:
                        images.append(reference_good.copy())
                return images
            # Trial path encodes page index as page-{index}.pptx
            name = Path(pptx_path).name
            if name.startswith("page-"):
                page_index = int(name.split("-")[1].split(".")[0])
                trial_pages.append(page_index)
            return [reference_good.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference_bad_target.copy()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch(
                    "pdf2ppt.refine.pptx_content_digest",
                    side_effect=lambda _p: next(digests, "done"),
                ):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.4):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                            _pages, report = refine_pages(
                                pages,
                                pdf_path=Path(tmp) / "src.pdf",
                                output_pptx=out,
                                slide_width_pt=64.0,
                                slide_height_pt=36.0,
                                iterations=1,
                                max_fixes_per_slide=1,
                                max_attempts_per_slide=2,
                                time_budget_seconds=25.0,
                                dpi=72,
                                write_pptx=fake_write,
                                render_pptx=fake_render_pptx,
                                render_pdf_page=fake_render_pdf,
                                work_dir=Path(tmp) / "work",
                            )

        self.assertEqual(len(report.pages), page_count)
        # Only pages with issues should be trialled; early indices first.
        self.assertTrue(trial_pages)
        self.assertLessEqual(max(trial_pages), 1)
        self.assertIn(0, trial_pages)
        # Late clean pages must not be marked time_budget solely due to ordering.
        self.assertIsNone(report.pages[-1].skipped_reason)

    def test_accepted_fix_resets_element_attempt_cap(self) -> None:
        """After an accept, the same element may try another action (widen then shrink)."""

        reference = _solid((120, 80), (255, 255, 255))
        ImageDraw.Draw(reference).text((12, 20), "Body text here", fill=(0, 0, 0))
        # Start slightly worse; each accepted trial improves SSIM toward the reference.
        baseline = _solid((120, 80), (255, 255, 255))
        ImageDraw.Draw(baseline).text((12, 20), "Body text here", fill=(90, 90, 90))
        after_widen = _solid((120, 80), (255, 255, 255))
        ImageDraw.Draw(after_widen).text((12, 20), "Body text here", fill=(40, 40, 40))
        after_shrink = reference.copy()

        page = _page(
            text_blocks=[
                TextBlock(
                    id="body",
                    source="native",
                    bbox=(10, 10, 60, 70),
                    text="Body text here",
                    confidence=1.0,
                    font_size=18.0,
                    line_bboxes=[(10, 10, 60, 25), (10, 28, 60, 43), (10, 46, 60, 61)],
                    paragraphs=[TextParagraph(text="Body text here", font_size=18.0)],
                )
            ],
            width=120.0,
            height=80.0,
        )
        call_state = {"n": 0}
        digests = iter([f"d{i}" for i in range(30)])
        issue = RefineIssue(
            kind="extra_wrap",
            element_id="body",
            severity=0.8,
            details={"element_kind": "text", "render_evidence": True, "source_lines": 2, "rendered_lines": 3},
        )

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            call_state["n"] += 1
            if call_state["n"] == 1:
                return [baseline.copy()]
            if call_state["n"] == 2:
                return [after_widen.copy()]
            return [after_shrink.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        diagnose_calls = {"n": 0}

        def fake_diagnose(reference_img, candidate_img, page_result, *, page_ssim=None, local_ssim_gap=0.02):
            diagnose_calls["n"] += 1
            # Keep surfacing the same body issue until two accepts land.
            metric = 0.70 + 0.05 * max(0, call_state["n"] - 1)
            if diagnose_calls["n"] <= 4:
                return metric, [issue]
            return metric, []

        apply_calls: list[str] = []
        propose_calls = {"n": 0}

        def fake_apply(page_arg, *, element_id, action, reference_page_image, **kwargs):
            apply_calls.append(action)
            # Mutate font/bbox so XML digests differ.
            block = page_arg.text_blocks[0]
            if action == "widen_bbox":
                x0, y0, x1, y1 = block.bbox
                block.bbox = (x0, y0, x1 + 4.0 + len(apply_calls), y1)
                return {"action": action, "element_id": element_id}
            if action == "shrink_font":
                block.font_size = float(block.font_size or 18.0) - 0.5
                return {"action": action, "element_id": element_id}
            return {"action": action, "element_id": element_id}

        def fake_propose(_issue: RefineIssue) -> list[str]:
            # First improving pass: widen only; after reset, allow shrink follow-up.
            propose_calls["n"] += 1
            if propose_calls["n"] <= 1:
                return ["widen_bbox"]
            return ["shrink_font"]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch(
                    "pdf2ppt.refine.pptx_content_digest",
                    side_effect=lambda _p: next(digests, "done"),
                ):
                    with patch("pdf2ppt.refine.diagnose_slide", side_effect=fake_diagnose):
                        with patch("pdf2ppt.refine.apply_fix_action", side_effect=fake_apply):
                            with patch(
                                "pdf2ppt.refine.propose_fixes_for_issue",
                                side_effect=fake_propose,
                            ):
                                with patch("pdf2ppt.refine.slide_editability_score", return_value=0.5):
                                    with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                                        _pages, report = refine_pages(
                                            [page],
                                            pdf_path=Path(tmp) / "src.pdf",
                                            output_pptx=out,
                                            slide_width_pt=120.0,
                                            slide_height_pt=80.0,
                                            iterations=2,
                                            max_fixes_per_slide=4,
                                            max_attempts_per_slide=8,
                                            max_attempts_per_element=1,
                                            dpi=72,
                                            write_pptx=fake_write,
                                            render_pptx=fake_render_pptx,
                                            render_pdf_page=fake_render_pdf,
                                            work_dir=Path(tmp) / "work",
                                        )

        accepted = [fix for fix in report.pages[0].fixes if fix.accepted]
        # Cap of 1 without reset would stop after widen; reset allows shrink follow-up.
        self.assertGreaterEqual(len(accepted), 2)
        self.assertEqual({fix.element_id for fix in accepted}, {"body"})
        self.assertIn("widen_bbox", [fix.action for fix in accepted])
        self.assertIn("shrink_font", [fix.action for fix in accepted])

    def test_default_max_attempts_is_eight(self) -> None:
        self.assertEqual(DEFAULT_MAX_ATTEMPTS_PER_SLIDE, 8)
        self.assertEqual(DEFAULT_MAX_ATTEMPTS_PER_ELEMENT, 2)

    def test_order_pages_for_refine_is_deterministic(self) -> None:
        ordered = order_pages_for_refine([(0, 0.2), (1, 0.9), (2, 0.9), (3, 0.0)])
        self.assertEqual(ordered, [1, 2, 0, 3])
        self.assertEqual(page_issue_severity([]), 0.0)
        # Without SSIM, score is normalised issue terms only (not raw sum).
        continuous = page_issue_severity(
            [
                RefineIssue(kind="a", element_id="x", severity=0.2),
                RefineIssue(kind="b", element_id="y", severity=0.7),
            ]
        )
        self.assertGreater(continuous, 0.0)
        self.assertLess(continuous, 2.0)

    def test_page_refine_severity_spreads_beyond_unit_clamp(self) -> None:
        """Worse fidelity and more issues must outrank saturated per-issue 1.0 ties."""

        mild = page_refine_severity(
            page_ssim=0.95,
            issues=[
                RefineIssue(
                    kind="low_local_ssim",
                    element_id="a",
                    severity=1.0,
                    details={"bbox": [0, 0, 10, 10], "element_kind": "shape"},
                )
            ],
            page_width_pt=100.0,
            page_height_pt=100.0,
        )
        severe = page_refine_severity(
            page_ssim=0.70,
            issues=[
                RefineIssue(
                    kind="low_local_ssim",
                    element_id="a",
                    severity=1.0,
                    details={"bbox": [0, 0, 40, 40], "element_kind": "shape"},
                ),
                RefineIssue(
                    kind="text_overflow",
                    element_id="t",
                    severity=1.0,
                    details={"bbox": [10, 10, 80, 30], "element_kind": "text"},
                ),
            ],
            page_width_pt=100.0,
            page_height_pt=100.0,
        )
        self.assertGreater(severe, mild)
        self.assertGreater(severe, 1.0)
        self.assertGreater(mild, 0.0)
        # Ordering must prefer the worse page even when both have severity≥1 issues.
        self.assertEqual(order_pages_for_refine([(0, mild), (1, severe)]), [1, 0])

    def test_page_refine_severity_fidelity_dominates_issue_count(self) -> None:
        """A worse-SSIM page sorts first even with fewer diagnosed issues."""

        worse_ssim_few = page_refine_severity(
            page_ssim=0.90,
            issues=[
                RefineIssue(
                    kind="low_local_ssim",
                    element_id="a",
                    severity=1.0,
                    details={"bbox": [0, 0, 20, 20], "element_kind": "shape"},
                )
            ],
            page_width_pt=100.0,
            page_height_pt=100.0,
        )
        better_ssim_many = page_refine_severity(
            page_ssim=0.97,
            issues=[
                RefineIssue(
                    kind="low_local_ssim",
                    element_id=f"s{i}",
                    severity=1.0,
                    details={"bbox": [0, 0, 10, 10], "element_kind": "shape"},
                )
                for i in range(12)
            ],
            page_width_pt=100.0,
            page_height_pt=100.0,
        )
        self.assertGreater(worse_ssim_few, better_ssim_many)
        self.assertEqual(
            order_pages_for_refine([(0, better_ssim_many), (1, worse_ssim_few)]),
            [1, 0],
        )

    def test_select_baseline_partial_under_short_budget(self) -> None:
        pages = [
            _page(
                shape_elements=[ShapeElement(id=f"s{i}", kind="rect", bbox=(10, 10, 40, 40), fill_color="#FF0000")]
                if i < 5
                else [],
                width=64.0,
                height=36.0,
            )
            for i in range(30)
        ]
        for i, page in enumerate(pages):
            page.page_number = i + 1
        indices, mode = select_baseline_page_indices(
            pages,
            remaining_budget=10.0,
            minimum_trial_seconds=2.5,
        )
        self.assertIn(mode, {"partial", "full"})
        self.assertTrue(indices)
        self.assertLessEqual(len(indices), 30)
        if mode == "partial":
            self.assertLess(len(indices), 30)
            # Highest prerank pages (shapes) should be preferred.
            self.assertTrue(any(i < 5 for i in indices))
        from pdf2ppt.refine import estimate_baseline_phase_seconds

        est = estimate_baseline_phase_seconds(len(indices))
        self.assertLessEqual(est + 2.5, 10.0 + 1e-9)

    def test_short_budget_partial_baseline_still_runs_trials(self) -> None:
        """30-page deck + 10s budget must not spend the whole budget on baseline only."""

        import time as time_module

        page_count = 30
        pages = [
            _page(
                shape_elements=[
                    ShapeElement(
                        id=f"card_{index}",
                        kind="rect",
                        bbox=(12, 12, 40, 40),
                        fill_color="#FF0000",
                    )
                ],
                width=64.0,
                height=36.0,
            )
            for index in range(page_count)
        ]
        for index, page in enumerate(pages):
            page.page_number = index + 1
            # Make early pages riskier for prerank when baseline is partial.
            if index >= 10:
                page.shape_elements = []

        reference = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(reference).rectangle((12, 12, 40, 40), fill=(0, 100, 200))
        bad = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(bad).rectangle((12, 12, 40, 40), fill=(200, 0, 0))

        render_meta: list[tuple[str, int]] = []
        digests = iter([f"d{i}" for i in range(200)])
        written_slide_counts: dict[str, int] = {}

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            path = Path(output_path)
            path.write_bytes(b"PK fake")
            count = len(pages_arg)
            written_slide_counts[path.name] = count
            written_slide_counts[str(path)] = count
            return path

        def timed_render(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            path = Path(pptx_path)
            name = path.name
            n = written_slide_counts.get(name) or written_slide_counts.get(str(path), 1)
            label = str(work_dir or "")
            if "baseline" in label or name in {"out.pptx", "baseline-partial.pptx"}:
                time_module.sleep(0.15 + 0.03 * n)
                kind = "partial" if name == "baseline-partial.pptx" else "full"
                render_meta.append((kind, n))
                return [bad.copy() for _ in range(n)]
            time_module.sleep(0.2)
            render_meta.append(("trial", 1))
            return [reference.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        # Patch estimator so 30-page full baseline is clearly too expensive for 10s.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch(
                "pdf2ppt.refine.estimate_baseline_render_seconds",
                side_effect=lambda n, **kw: 2.0 + 0.25 * float(n),
            ):
                with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                    with patch(
                        "pdf2ppt.refine.pptx_content_digest",
                        side_effect=lambda _p: next(digests, "done"),
                    ):
                        with patch("pdf2ppt.refine.slide_editability_score", return_value=0.4):
                            with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                                _pages, report = refine_pages(
                                    pages,
                                    pdf_path=Path(tmp) / "src.pdf",
                                    output_pptx=out,
                                    slide_width_pt=64.0,
                                    slide_height_pt=36.0,
                                    iterations=1,
                                    max_fixes_per_slide=1,
                                    max_attempts_per_slide=2,
                                    time_budget_seconds=10.0,
                                    dpi=72,
                                    write_pptx=fake_write,
                                    render_pptx=timed_render,
                                    render_pdf_page=fake_render_pdf,
                                    work_dir=Path(tmp) / "work",
                                )

        self.assertEqual(report.baseline_mode, "partial")
        self.assertIsNotNone(report.baseline_page_count)
        assert report.baseline_page_count is not None
        self.assertLess(report.baseline_page_count, page_count)
        self.assertTrue(report.pages_skipped)
        self.assertTrue(all(entry["reason"] == "time_budget" for entry in report.pages_skipped))
        # At least one trial should have run (not baseline-only).
        trial_calls = [entry for entry in render_meta if entry[0] == "trial"]
        self.assertTrue(trial_calls)
        tried_pages = [page for page in report.pages if page.attempt_count > 0 or page.accepted_fix_count > 0]
        self.assertTrue(tried_pages)
        self.assertLessEqual(float(report.runtime_seconds or 0.0), 10.0 + 2.0)

    def test_reallocation_gives_late_page_borrowed_time(self) -> None:
        """Early pages finishing fast must free budget so a late page can accept a fix."""

        import time as time_module

        page_count = 6
        pages = [
            _page(
                shape_elements=[
                    ShapeElement(
                        id=f"card_{index}",
                        kind="rect",
                        bbox=(12, 12, 40, 40),
                        fill_color="#FF0000",
                    )
                ],
                width=64.0,
                height=36.0,
            )
            for index in range(page_count)
        ]
        for index, page in enumerate(pages):
            page.page_number = index + 1

        reference = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(reference).rectangle((12, 12, 40, 40), fill=(0, 100, 200))
        bad = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(bad).rectangle((12, 12, 40, 40), fill=(200, 0, 0))

        # Force late page to look worst so first pass visits it last; early pages
        # exhaust candidates quickly (noop digests), leaving wall time for pass 2.
        digests = iter([f"d{i}" for i in range(200)])
        call_state = {"n": 0}
        late_accepts = {"n": 0}
        written_counts: dict[str, int] = {}

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            path = Path(output_path)
            path.write_bytes(b"PK fake")
            written_counts[path.name] = len(pages_arg)
            return path

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            call_state["n"] += 1
            name = Path(pptx_path).name
            n = written_counts.get(name, 1)
            if "baseline" in str(work_dir or "") or name in {"out.pptx", "baseline-partial.pptx"}:
                time_module.sleep(0.05)
                return [bad.copy() for _ in range(n)]
            # Slow trials so first-pass fair share (~budget/pages) fits ~1 attempt.
            time_module.sleep(1.2)
            if name.startswith("page-"):
                page_index = int(name.split("-")[1].split(".")[0])
                if page_index == page_count - 1:
                    late_accepts["n"] += 1
                    # Needs a second attempt (reallocation round-robin) to accept.
                    if late_accepts["n"] >= 2:
                        return [reference.copy()]
            return [bad.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        # Early pages look worse (visited first); last page is next-worst so it
        # still has untried candidates when leftover time is reallocated.
        from pdf2ppt.refine import diagnose_slide as real_diagnose

        def fake_diagnose(reference_img, candidate_img, page_result, *, page_ssim=None, local_ssim_gap=0.02):
            metric, issues = real_diagnose(
                reference_img,
                candidate_img,
                page_result,
                page_ssim=page_ssim,
                local_ssim_gap=local_ssim_gap,
            )
            if not issues:
                issues = [
                    RefineIssue(
                        kind="low_local_ssim",
                        element_id=page_result.shape_elements[0].id,
                        severity=1.0,
                        details={"element_kind": "shape", "render_evidence": True, "bbox": [12, 12, 40, 40]},
                    )
                ]
            if page_result.page_number == 1:
                metric = 0.50
            elif page_result.page_number == page_count:
                metric = 0.62
            else:
                metric = 0.88
            return metric, issues

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch(
                    "pdf2ppt.refine.pptx_content_digest",
                    side_effect=lambda _p: next(digests, "done"),
                ):
                    with patch("pdf2ppt.refine.diagnose_slide", side_effect=fake_diagnose):
                        with patch("pdf2ppt.refine.slide_editability_score", return_value=0.4):
                            with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                                _pages, report = refine_pages(
                                    pages,
                                    pdf_path=Path(tmp) / "src.pdf",
                                    output_pptx=out,
                                    slide_width_pt=64.0,
                                    slide_height_pt=36.0,
                                    iterations=2,
                                    max_fixes_per_slide=2,
                                    max_attempts_per_slide=4,
                                    time_budget_seconds=20.0,
                                    dpi=72,
                                    write_pptx=fake_write,
                                    render_pptx=fake_render_pptx,
                                    render_pdf_page=fake_render_pdf,
                                    work_dir=Path(tmp) / "work",
                                )

        late = report.pages[-1]
        self.assertGreaterEqual(report.reallocation_passes, 1)
        self.assertGreater(late.accepted_fix_count, 0)
        self.assertGreater(float(late.borrowed_seconds or 0.0), 0.0)
        self.assertIsNotNone(late.severity)
        self.assertIsNotNone(late.time_used_seconds)
        # Used time must not exceed granted.
        if late.granted_seconds is not None and late.time_used_seconds is not None:
            self.assertLessEqual(late.time_used_seconds, late.granted_seconds + 0.5)

    def test_page_time_allowance_never_starves_usable_slot(self) -> None:
        floor = float(DEFAULT_MIN_SOFFICE_TIMEOUT_SECONDS)
        # 90/30 → fair 3s ≥ floor 2.5 → return fair share.
        self.assertAlmostEqual(
            page_time_allowance_seconds(90.0, pages_left=30, minimum_seconds=floor),
            3.0,
        )
        # 60/30 → fair 2s < floor → bump to floor (non-capped path).
        self.assertEqual(
            page_time_allowance_seconds(60.0, pages_left=30, minimum_seconds=floor),
            floor,
        )
        # With cap_at_fair_share: never inflate above equal share (first pass).
        self.assertAlmostEqual(
            page_time_allowance_seconds(60.0, pages_left=30, minimum_seconds=floor, cap_at_fair_share=True),
            2.0,
        )
        self.assertEqual(
            page_time_allowance_seconds(2.0, pages_left=10, minimum_seconds=floor),
            0.0,
        )
        self.assertAlmostEqual(
            page_time_allowance_seconds(60.0, pages_left=4, minimum_seconds=floor),
            15.0,
        )

    def test_noops_do_not_count_toward_attempt_cap(self) -> None:
        """XML-unchanged trials must not burn max_attempts_per_slide."""

        reference = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(reference).text((8, 8), "Title", fill=(0, 0, 0))
        same = reference.copy()

        page = _page(
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(5, 5, 55, 22),
                    text="Title",
                    confidence=1.0,
                    font_size=40.0,
                    line_bboxes=[(5, 5, 55, 22)],
                )
            ],
            width=64.0,
            height=36.0,
        )

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            return [same.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                # Digests always equal → every trial is a no-op.
                with patch("pdf2ppt.refine.pptx_content_digest", return_value="same"):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.5):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                            _pages, report = refine_pages(
                                [page],
                                pdf_path=Path(tmp) / "src.pdf",
                                output_pptx=out,
                                slide_width_pt=64.0,
                                slide_height_pt=36.0,
                                iterations=2,
                                max_fixes_per_slide=4,
                                max_attempts_per_slide=2,
                                dpi=72,
                                write_pptx=fake_write,
                                render_pptx=fake_render_pptx,
                                render_pdf_page=fake_render_pdf,
                                work_dir=Path(tmp) / "work",
                            )

        slide = report.pages[0]
        noops = [fix for fix in slide.fixes if fix.noop]
        self.assertTrue(noops)
        self.assertEqual(slide.attempt_count, 0)
        self.assertEqual(report.attempt_count, 0)

    def test_per_element_attempt_cap_lets_other_elements_try(self) -> None:
        reference = _solid((160, 90), (255, 255, 255))
        ImageDraw.Draw(reference).rectangle((12, 12, 36, 36), fill=(0, 100, 200))
        ImageDraw.Draw(reference).rectangle((80, 12, 104, 36), fill=(0, 150, 50))
        bad = reference.copy()
        ImageDraw.Draw(bad).rectangle((12, 12, 36, 36), fill=(200, 0, 0))
        ImageDraw.Draw(bad).rectangle((80, 12, 104, 36), fill=(200, 0, 0))

        page = _page(
            shape_elements=[
                ShapeElement(id="card_a", kind="rect", bbox=(12, 12, 36, 36), fill_color="#FF0000"),
                ShapeElement(id="card_b", kind="rect", bbox=(80, 12, 104, 36), fill_color="#FF0000"),
            ],
            width=160.0,
            height=90.0,
        )
        call_state = {"n": 0}
        digests = iter([f"d{i}" for i in range(20)])

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            Path(output_path).write_bytes(b"PK fake")
            return Path(output_path)

        def fake_render_pptx(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            # Always return the bad image so no fix is accepted; both elements
            # should still receive capped attempts.
            call_state["n"] += 1
            return [bad.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch(
                    "pdf2ppt.refine.pptx_content_digest",
                    side_effect=lambda _p: next(digests, "done"),
                ):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.4):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                            _pages, report = refine_pages(
                                [page],
                                pdf_path=Path(tmp) / "src.pdf",
                                output_pptx=out,
                                slide_width_pt=160.0,
                                slide_height_pt=90.0,
                                iterations=2,
                                max_fixes_per_slide=4,
                                max_attempts_per_slide=8,
                                max_attempts_per_element=2,
                                dpi=72,
                                write_pptx=fake_write,
                                render_pptx=fake_render_pptx,
                                render_pdf_page=fake_render_pdf,
                                work_dir=Path(tmp) / "work",
                            )

        elements_tried = {fix.element_id for fix in report.pages[0].fixes if fix.xml_changed}
        self.assertIn("card_a", elements_tried)
        self.assertIn("card_b", elements_tried)
        for element_id in ("card_a", "card_b"):
            real = [fix for fix in report.pages[0].fixes if fix.element_id == element_id and fix.xml_changed]
            self.assertLessEqual(len(real), DEFAULT_MAX_ATTEMPTS_PER_ELEMENT)

    def test_slow_soffice_marks_time_budget_not_render_failed(self) -> None:
        """Fake 2s soffice: pages beyond budget are time_budget, never false render_failed."""

        import time as time_module

        pages = [
            _page(
                shape_elements=[
                    ShapeElement(
                        id=f"card_{index}",
                        kind="rect",
                        bbox=(12, 12, 40, 40),
                        fill_color="#FF0000",
                    )
                ],
                width=64.0,
                height=36.0,
            )
            for index in range(4)
        ]
        for index, page in enumerate(pages):
            page.page_number = index + 1

        reference = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(reference).rectangle((12, 12, 40, 40), fill=(0, 100, 200))
        bad = _solid((64, 36), (255, 255, 255))
        ImageDraw.Draw(bad).rectangle((12, 12, 40, 40), fill=(200, 0, 0))

        timeouts: list[int] = []
        digests = iter([f"d{i}" for i in range(40)])
        written_counts: dict[str, int] = {}

        def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
            path = Path(output_path)
            path.write_bytes(b"PK fake")
            written_counts[path.name] = len(pages_arg)
            return path

        def slow_render(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
            timeouts.append(int(timeout_seconds))
            # Mimic a real convert (~2s) without hanging forever.
            sleep_for = min(2.0, max(0.0, float(timeout_seconds) - 0.05))
            time_module.sleep(sleep_for)
            name = Path(pptx_path).name
            n = written_counts.get(name, 1)
            if "baseline" in str(work_dir or "") or name in {"out.pptx", "baseline-partial.pptx"}:
                return [bad.copy() for _ in range(n)]
            return [bad.copy()]

        def fake_render_pdf(pdf_path, page_index, *, dpi=144):
            return reference.copy()

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.pptx"
            (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
            with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
                with patch(
                    "pdf2ppt.refine.pptx_content_digest",
                    side_effect=lambda _p: next(digests, "done"),
                ):
                    with patch("pdf2ppt.refine.slide_editability_score", return_value=0.5):
                        with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                            _pages, report = refine_pages(
                                pages,
                                pdf_path=Path(tmp) / "src.pdf",
                                output_pptx=out,
                                slide_width_pt=64.0,
                                slide_height_pt=36.0,
                                iterations=1,
                                max_fixes_per_slide=1,
                                max_attempts_per_slide=2,
                                time_budget_seconds=7.0,
                                dpi=72,
                                write_pptx=fake_write,
                                render_pptx=slow_render,
                                render_pdf_page=fake_render_pdf,
                                work_dir=Path(tmp) / "work",
                            )

        self.assertIsNone(report.skipped_reason)
        self.assertEqual(len(report.pages), 4)
        reasons = [page.skipped_reason for page in report.pages]
        self.assertNotIn("render_failed", reasons)
        self.assertTrue(any(reason == "time_budget" for reason in reasons))
        # Never issue a trial timeout below the minimum floor.
        self.assertTrue(timeouts)
        self.assertTrue(all(value >= int(min_soffice_timeout_seconds(None)) for value in timeouts))

    def test_write_refine_report_json(self) -> None:
        report = DeckRefineReport(
            enabled=True,
            skipped_reason=None,
            iterations=DEFAULT_REFINE_ITERS,
            dpi=144,
            pages=[],
            runtime_seconds=0.1,
            soffice_call_count=3,
            attempt_count=5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_refine_report(report, Path(tmp) / "out.refine.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["soffice_call_count"], 3)
        self.assertEqual(payload["attempt_count"], 5)


class RefineCliValidationTests(unittest.TestCase):
    def test_refine_time_budget_rejects_negative(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["in.pdf", "out.pptx", "--refine-time-budget", "-5"])

    def test_refine_time_budget_rejects_zero(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["in.pdf", "out.pptx", "--refine-time-budget", "0"])

    def test_refine_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["in.pdf", "out.pptx", "--refine"])
        self.assertEqual(args.refine_max_attempts, 8)
        self.assertEqual(args.refine_min_font_size, 8.0)
        self.assertEqual(args.refine_min_font_ratio, 0.95)
        self.assertEqual(args.refine_time_budget, 90.0)


@unittest.skipUnless(find_libreoffice_executable() is not None, "LibreOffice not installed")
@unittest.skipUnless(FIXTURES_DIR.joinpath("solid_background.pdf").exists(), "fixtures missing")
class RefineLibreOfficeIntegrationTests(unittest.TestCase):
    def test_refine_smoke_never_regresses_ssim(self) -> None:
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
                    paragraphs=[TextParagraph(text="Solid Background", font_size=28.0)],
                )
            ],
            width=960.0,
            height=540.0,
        )
        import io

        buffer_img = Image.new("RGB", (96, 54), color=(245, 248, 252))
        buf = io.BytesIO()
        buffer_img.save(buf, format="PNG")
        page.background_image_bytes = buf.getvalue()
        page.background_mode = "full-page"

        with tempfile.TemporaryDirectory() as tmp:
            pptx_path = Path(tmp) / "deck.pptx"
            render_pages_to_pptx([page], pptx_path, slide_width_pt=960.0, slide_height_pt=540.0)
            before = evaluate_pptx_against_pdf(pptx_path, pdf_path, dpi=72, ocr_engine=None)
            before_digest = pptx_content_digest(pptx_path)
            _pages, report = refine_pages(
                [page],
                pdf_path=pdf_path,
                output_pptx=pptx_path,
                slide_width_pt=960.0,
                slide_height_pt=540.0,
                iterations=2,
                max_fixes_per_slide=4,
                dpi=72,
                work_dir=Path(tmp) / "refine",
            )
            after = evaluate_pptx_against_pdf(pptx_path, pdf_path, dpi=72, ocr_engine=None)
            after_digest = pptx_content_digest(pptx_path)

        self.assertNotEqual(report.skipped_reason, "libreoffice_unavailable")
        self.assertEqual(len(report.pages), 1)
        self.assertGreaterEqual(
            after.pages[0].fidelity.ssim,
            before.pages[0].fidelity.ssim - 1e-6,
        )
        # Neutral accepts are gone: either digest unchanged, or SSIM strictly improved.
        if after_digest != before_digest:
            self.assertGreater(
                after.pages[0].fidelity.ssim,
                before.pages[0].fidelity.ssim + METRIC_IMPROVE_EPSILON - 1e-9,
            )
        accepted = [fix for fix in report.pages[0].fixes if fix.accepted]
        for fix in accepted:
            self.assertFalse(fix.noop)
            self.assertTrue(fix.xml_changed)
            assert fix.metric_after is not None
            self.assertGreater(fix.metric_after, fix.metric_before + METRIC_IMPROVE_EPSILON - 1e-9)


@unittest.skipUnless(find_libreoffice_executable() is not None, "LibreOffice not installed")
class RefineProbeLibreOfficeTests(unittest.TestCase):
    """Targeted probes for defects that previously burned the refine budget."""

    def _roundtrip_shape_probe(self, *, fill_ref: tuple[int, int, int], fill_bad: tuple[int, int, int]) -> None:
        """Shape recolor defect should prefer raster_fallback and may improve SSIM."""

        from pdf2ppt.pipeline import render_pages_to_pptx

        reference = _solid((320, 180), (255, 255, 255))
        ImageDraw.Draw(reference).rectangle((40, 40, 160, 120), fill=fill_ref)
        # Build a PDF-like page result with the wrong fill.
        page = _page(
            shape_elements=[
                ShapeElement(
                    id="blend_card",
                    kind="rect",
                    bbox=(40, 40, 160, 120),
                    fill_color="#{:02X}{:02X}{:02X}".format(*fill_bad),
                )
            ],
            width=320.0,
            height=180.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            # Synthetic "source PDF" page image used as reference via monkeypatch.
            pdf_path = Path(tmp) / "probe.pdf"
            # Minimal valid-enough PDF is not required: we inject render_pdf_page.
            pdf_path.write_bytes(b"%PDF-1.4\n")
            pptx_path = Path(tmp) / "probe.pptx"
            render_pages_to_pptx([page], pptx_path, slide_width_pt=320.0, slide_height_pt=180.0)

            def fake_pdf(_path, _index, *, dpi=144):
                return reference.copy()

            _pages, report = refine_pages(
                [page],
                pdf_path=pdf_path,
                output_pptx=pptx_path,
                slide_width_pt=320.0,
                slide_height_pt=180.0,
                iterations=2,
                max_fixes_per_slide=4,
                dpi=72,
                render_pdf_page=fake_pdf,
                work_dir=Path(tmp) / "refine",
            )

        self.assertEqual(len(report.pages), 1)
        # Diagnosis should surface the shape (render evidence), not a title wrap.
        kinds = [issue.kind for issue in report.pages[0].issues]
        self.assertIn("low_local_ssim", kinds)
        # If anything is accepted, it must be raster_fallback on the shape.
        for fix in report.pages[0].fixes:
            if fix.accepted:
                self.assertEqual(fix.action, "raster_fallback")
                self.assertEqual(fix.element_id, "blend_card")

    def test_multiply_blend_probe_targets_shape_not_title(self) -> None:
        self._roundtrip_shape_probe(fill_ref=(30, 90, 180), fill_bad=(0, 0, 255))


if __name__ == "__main__":
    unittest.main()
