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

from pdf2ppt.models import (
    ImagePlacement,
    PageResult,
    QualityScore,
    ShapeElement,
    TextBlock,
    TextParagraph,
)
from pdf2ppt.refine import (
    DEFAULT_REFINE_ITERS,
    METRIC_IMPROVE_EPSILON,
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
    pptx_content_digest,
    propose_fixes_for_issue,
    refine_pages,
    region_ssim,
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
        details = apply_shrink_font(page, "t", step_pt=1.0)
        self.assertIsNotNone(details)
        assert details is not None
        self.assertTrue(details["paragraph_sizes_scaled"])
        self.assertEqual(page.text_blocks[0].font_size, 17.0)
        paras = page.text_blocks[0].paragraphs or []
        self.assertEqual(paras[0].font_size, 17.0)
        self.assertEqual(paras[1].font_size, 15.0)

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
