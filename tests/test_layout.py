"""P2 layout analysis tests — heuristic path + mocked PP-StructureV3."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
from pptx import Presentation

from pdf2ppt.core import ConversionOptions
from pdf2ppt.layout import (
    LayoutEngine,
    LayoutPageResult,
    LayoutRegion,
    analyze_layout_heuristic,
    apply_layout_to_page,
    normalize_layout_label,
    parse_layout_regions_from_dict,
    strip_bullet_prefix,
)
from pdf2ppt.layout_metrics import (
    count_pptx_bullet_paragraphs,
    evaluate_text_grouping,
    load_layout_ground_truth,
    slide_has_title_placeholder_text,
)
from pdf2ppt.models import TextBlock, TextParagraph
from pdf2ppt.ppt_render import add_text_block, render_page_to_slide

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _block(
    block_id: str,
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    role: str = "body",
    font_size: float = 14.0,
) -> TextBlock:
    return TextBlock(
        id=block_id,
        source="native",
        bbox=bbox,
        text=text,
        confidence=0.99,
        font_size=font_size,
        block_role=role,
    )


class NormalizeLabelTests(unittest.TestCase):
    def test_maps_ppstructure_labels(self) -> None:
        self.assertEqual(normalize_layout_label("doc_title"), "title")
        self.assertEqual(normalize_layout_label("figure_title"), "caption")
        self.assertEqual(normalize_layout_label("page_number"), "page_number")
        self.assertEqual(normalize_layout_label("unknown_xyz"), "other")


class HeuristicLayoutTests(unittest.TestCase):
    def test_keeps_flowchart_labels_separate(self) -> None:
        blocks = [
            _block("t", "Onboarding Flow", (40, 24, 600, 70), role="title", font_size=24),
            _block("a", "Collect Info", (128, 268, 312, 312), font_size=16),
            _block("b", "Verify", (528, 268, 752, 312), font_size=16),
        ]
        regions = analyze_layout_heuristic(blocks, page_width=960, page_height=540)
        applied = apply_layout_to_page(
            blocks,
            LayoutPageResult(regions=regions, engine="heuristic"),
            page_width=960,
            page_height=540,
        )
        texts = {block.text for block in applied.text_blocks}
        self.assertIn("Collect Info", texts)
        self.assertIn("Verify", texts)
        merged = [block for block in applied.text_blocks if "Collect" in block.text and "Verify" in block.text]
        self.assertEqual(merged, [])

    def test_groups_stacked_list_lines(self) -> None:
        blocks = [
            _block("t", "Sprint Goals", (48, 36, 400, 80), role="title", font_size=28),
            _block("l1", "• Ship layout analysis", (72, 130, 700, 160)),
            _block("l2", "◦ Detect multi-level bullets", (96, 170, 700, 200)),
            _block("l3", "• Keep heuristic fallback", (72, 250, 700, 280)),
        ]
        regions = analyze_layout_heuristic(blocks, page_width=960, page_height=540)
        applied = apply_layout_to_page(
            blocks,
            LayoutPageResult(regions=regions, engine="heuristic"),
            page_width=960,
            page_height=540,
            skip_header_footer=True,
        )
        list_blocks = [block for block in applied.text_blocks if block.block_role == "list"]
        self.assertTrue(list_blocks)
        paragraphs = list_blocks[0].paragraphs or []
        self.assertGreaterEqual(len(paragraphs), 2)
        self.assertTrue(any(para.list_level is not None for para in paragraphs))
        # Literal bullet glyphs stripped from paragraph text.
        self.assertFalse(any(para.text.startswith(("•", "◦", "-", "*")) for para in paragraphs))

    def test_skips_footer_and_page_number(self) -> None:
        blocks = [
            _block("body", "Hello", (60, 120, 400, 160)),
            _block("foot", "Confidential", (48, 500, 300, 530), font_size=10),
            _block("num", "3", (880, 500, 940, 530), font_size=10),
        ]
        regions = analyze_layout_heuristic(blocks, page_width=960, page_height=540)
        applied = apply_layout_to_page(
            blocks,
            LayoutPageResult(regions=regions, engine="heuristic"),
            page_width=960,
            page_height=540,
            skip_header_footer=True,
        )
        texts = [block.text for block in applied.text_blocks]
        self.assertIn("Hello", texts)
        self.assertNotIn("Confidential", texts)
        self.assertNotIn("3", texts)

    def test_strip_bullet_prefix_levels(self) -> None:
        cleaned, level = strip_bullet_prefix("• Ship it", left=72, min_left=72)
        self.assertEqual(cleaned, "Ship it")
        self.assertEqual(level, 0)
        cleaned, level = strip_bullet_prefix("1. First", left=96, min_left=72)
        self.assertEqual(cleaned, "First")
        self.assertGreaterEqual(level, 1)


class PpstructureMockTests(unittest.TestCase):
    def test_ppstructure_parses_mocked_boxes(self) -> None:
        image = Image.new("RGB", (960, 540), color=(255, 255, 255))
        fake_boxes = [
            {"cls_id": 0, "label": "doc_title", "score": 0.99, "coordinate": [48, 36, 500, 90]},
            {"cls_id": 2, "label": "text", "score": 0.95, "coordinate": [60, 140, 450, 400]},
            {"cls_id": 2, "label": "text", "score": 0.94, "coordinate": [510, 140, 900, 400]},
            {"cls_id": 4, "label": "figure", "score": 0.91, "coordinate": [180, 110, 780, 380]},
            {"cls_id": 6, "label": "figure_title", "score": 0.88, "coordinate": [180, 395, 780, 440]},
            {"cls_id": 5, "label": "footer", "score": 0.8, "coordinate": [40, 500, 200, 530]},
        ]

        class FakeResult:
            def __init__(self) -> None:
                self.json = {"res": {"boxes": fake_boxes}}

        class FakeDetector:
            def predict(self, *_args, **_kwargs):
                return [FakeResult()]

        engine = LayoutEngine(engine="ppstructure")
        with patch.object(LayoutEngine, "_get_detector", return_value=FakeDetector()):
            result = engine.analyze(image, page_width=960, page_height=540, page_number=1)

        self.assertEqual(result.engine, "ppstructure")
        labels = {region.label for region in result.regions}
        self.assertIn("title", labels)
        self.assertIn("figure", labels)
        self.assertIn("caption", labels)
        self.assertIn("footer", labels)
        # Image coords map 1:1 when image size matches page points.
        title = next(region for region in result.regions if region.label == "title")
        self.assertAlmostEqual(title.bbox[0], 48, places=0)

    def test_ppstructure_falls_back_when_detector_fails(self) -> None:
        blocks = [_block("a", "Hello", (40, 40, 200, 80))]
        engine = LayoutEngine(engine="ppstructure")
        with patch.object(LayoutEngine, "_get_detector", side_effect=RuntimeError("no paddle")):
            result = engine.analyze(
                None,
                page_width=960,
                page_height=540,
                text_blocks=blocks,
            )
        self.assertEqual(result.engine, "heuristic")
        self.assertIsNotNone(result.fallback_reason)

    def test_parse_layout_regions_from_dict(self) -> None:
        regions = parse_layout_regions_from_dict(
            [{"label": "list", "bbox": [10, 20, 100, 200], "score": 0.9}],
            page_number=2,
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].label, "list")
        self.assertEqual(regions[0].reading_order, 1)


class ApplyLayoutFigureTableTests(unittest.TestCase):
    def test_crops_figure_regions(self) -> None:
        import numpy as np

        array = np.full((540, 960, 3), (30, 90, 160), dtype=np.uint8)
        array[110:380, 180:780] = (200, 180, 40)
        image = Image.fromarray(array, mode="RGB")
        regions = [
            LayoutRegion(
                id="fig1",
                label="figure",
                bbox=(180, 110, 780, 380),
                confidence=0.95,
                reading_order=1,
                engine="ppstructure",
            )
        ]
        applied = apply_layout_to_page(
            [],
            LayoutPageResult(regions=regions, engine="ppstructure"),
            page_width=960,
            page_height=540,
            page_image=image,
        )
        self.assertEqual(len(applied.figure_images), 1)
        self.assertTrue(applied.figure_images[0].png_bytes.startswith(b"\x89PNG"))

    def test_defers_text_inside_native_table(self) -> None:
        from pdf2ppt.models import TableCell, TableElement

        table = TableElement(
            id="t1",
            bbox=(100, 100, 500, 300),
            row_count=1,
            col_count=1,
            cells=[TableCell(row=0, col=0, text="Cell", text_block_ids=["n1"])],
            column_widths=[400],
            row_heights=[200],
        )
        blocks = [_block("n1", "Cell", (120, 120, 400, 180))]
        regions = [
            LayoutRegion(
                id="table",
                label="table",
                bbox=(100, 100, 500, 300),
                confidence=0.9,
                reading_order=1,
            )
        ]
        applied = apply_layout_to_page(
            blocks,
            LayoutPageResult(regions=regions, engine="heuristic"),
            page_width=960,
            page_height=540,
            table_elements=[table],
        )
        self.assertEqual(applied.text_blocks, [])
        self.assertIn("n1", applied.skipped_block_ids)


class BulletRenderTests(unittest.TestCase):
    def test_set_paragraph_bullet_emits_bu_char(self) -> None:
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        block = TextBlock(
            id="list1",
            source="native",
            bbox=(72, 120, 600, 280),
            text="One\nTwo",
            confidence=1.0,
            font_size=16,
            block_role="list",
            paragraphs=[
                TextParagraph(text="One", list_level=0),
                TextParagraph(text="Two", list_level=1),
            ],
        )
        add_text_block(slide, block, scale_x=1.0, scale_y=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bullets.pptx"
            presentation.save(path)
            self.assertGreaterEqual(count_pptx_bullet_paragraphs(path), 2)

    def test_title_placeholder_mapping(self) -> None:
        from pdf2ppt.models import PageResult, QualityScore

        presentation = Presentation()
        blank = presentation.slide_layouts[6]
        page = PageResult(
            page_number=1,
            page_kind="digital",
            background_mode="elements",
            width_pt=960,
            height_pt=540,
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(60, 40, 700, 100),
                    text="Platform Overview",
                    confidence=1.0,
                    font_size=28,
                    block_role="title",
                    use_title_placeholder=True,
                    reading_order=1,
                ),
                TextBlock(
                    id="body",
                    source="native",
                    bbox=(60, 140, 400, 300),
                    text="Body copy",
                    confidence=1.0,
                    font_size=14,
                    reading_order=2,
                ),
            ],
            quality_score=QualityScore(1, 1, 1, 1),
            fallback_reason=None,
        )
        render_page_to_slide(
            presentation,
            blank,
            page,
            slide_width_pt=960,
            slide_height_pt=540,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "title.pptx"
            presentation.save(path)
            self.assertTrue(slide_has_title_placeholder_text(path))


class LayoutMetricsFixtureTests(unittest.TestCase):
    def test_flowchart_grouping_metrics_no_merge(self) -> None:
        gt = load_layout_ground_truth(FIXTURES / "flowchart.layout.json")
        blocks = [
            _block("t", "Onboarding Flow", (40, 24, 600, 70), role="title", font_size=24),
            _block("s", "Start", (128, 138, 312, 182), font_size=16),
            _block("c", "Collect Info", (128, 268, 312, 312), font_size=16),
            _block("v", "Verify", (528, 268, 752, 312), font_size=16),
            _block("d", "Done", (328, 418, 552, 462), font_size=16),
        ]
        regions = analyze_layout_heuristic(blocks, page_width=960, page_height=540)
        applied = apply_layout_to_page(
            blocks,
            LayoutPageResult(regions=regions, engine="heuristic"),
            page_width=960,
            page_height=540,
        )
        metrics = evaluate_text_grouping(applied.text_blocks, gt)
        self.assertEqual(metrics.merge_error_count, 0)
        self.assertTrue(metrics.title_correct)
        self.assertGreaterEqual(metrics.mean_paragraph_iou, 0.35)

    def test_bullet_list_fixture_metrics(self) -> None:
        gt = load_layout_ground_truth(FIXTURES / "bullet_list.layout.json")
        blocks = [
            _block("t", "Sprint Goals", (48, 36, 800, 90), role="title", font_size=28),
            _block("a", "• Ship layout analysis for scanned decks", (72, 130, 880, 164)),
            _block("b", "◦ Detect multi-level bullets", (96, 170, 880, 204)),
            _block("c", "◦ Map titles to placeholders", (96, 205, 880, 239)),
            _block("d", "• Keep heuristic fallback for CI", (72, 250, 880, 284)),
            _block("e", "◦ No model download required", (96, 290, 880, 324)),
            _block("f", "• Report grouping metrics in the PR", (72, 340, 880, 374)),
            _block("foot", "Confidential", (48, 500, 400, 530), font_size=10),
            _block("num", "3", (880, 500, 940, 530), font_size=10),
        ]
        regions = analyze_layout_heuristic(blocks, page_width=960, page_height=540)
        applied = apply_layout_to_page(
            blocks,
            LayoutPageResult(regions=regions, engine="heuristic"),
            page_width=960,
            page_height=540,
            skip_header_footer=True,
        )
        metrics = evaluate_text_grouping(applied.text_blocks, gt)
        self.assertTrue(metrics.title_correct)
        self.assertIsNotNone(metrics.list_f1)
        assert metrics.list_f1 is not None
        self.assertGreaterEqual(metrics.list_f1, 0.5)
        # Footer/page number should not appear as predicted boxes.
        joined = " ".join(block.text for block in applied.text_blocks)
        self.assertNotIn("Confidential", joined)


class PipelineLayoutOptionTests(unittest.TestCase):
    def test_conversion_options_default_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            options = ConversionOptions(
                input_path=Path(tmp) / "in.pdf",
                output_path=Path(tmp) / "out.pptx",
                report_path=Path(tmp) / "out.json",
            )
        self.assertEqual(options.layout_engine, "heuristic")
        self.assertTrue(options.layout_skip_header_footer)

    def test_analyze_page_records_layout_engine(self) -> None:
        from pdf2ppt.pipeline import analyze_page

        pdf_path = FIXTURES / "flowchart.pdf"
        import fitz

        options = ConversionOptions(
            input_path=pdf_path,
            output_path=pdf_path.with_suffix(".pptx"),
            report_path=pdf_path.with_suffix(".json"),
            vector_mode="off",
            layout_engine="heuristic",
        )
        ocr = MagicMock()
        with fitz.open(pdf_path) as document:
            page = document[0]
            result = analyze_page(page, options, ocr)
        self.assertEqual(result.layout_engine, "heuristic")
        self.assertTrue(result.layout_regions)
        texts = [block.text for block in result.text_blocks]
        self.assertTrue(any("Collect Info" in text for text in texts))
        self.assertTrue(any("Verify" in text for text in texts))
        self.assertFalse(any("Collect Info" in text and "Verify" in text for text in texts))


class RealModelOptInTests(unittest.TestCase):
    """Real PP-StructureV3 download is opt-in via PDF2PPT_RUN_LAYOUT_MODEL=1."""

    def test_real_model_skipped_by_default(self) -> None:
        import os

        if os.environ.get("PDF2PPT_RUN_LAYOUT_MODEL") != "1":
            self.skipTest("Set PDF2PPT_RUN_LAYOUT_MODEL=1 to run the real layout model")
        from pdf2ppt.layout import is_ppstructure_available

        if not is_ppstructure_available():
            self.skipTest("LayoutDetection not importable")
        image = Image.new("RGB", (640, 480), color=(255, 255, 255))
        engine = LayoutEngine(engine="ppstructure")
        result = engine.analyze(image, page_width=640, page_height=480)
        self.assertIn(result.engine, {"ppstructure", "heuristic"})


if __name__ == "__main__":
    unittest.main()
