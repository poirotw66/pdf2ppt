from __future__ import annotations

import unittest

import fitz
from PIL import Image

from pdf2ppt.cli import build_parser
from pdf2ppt.models import QualityScore, TextBlock
from pdf2ppt.pipeline import (
    build_text_fit_debug_entry,
    build_mask_shapes,
    classify_text_script,
    estimate_font_size,
    PageSignals,
    choose_background_mode,
    classify_page,
    intersection_ratio,
    mask_text_regions_with_white_boxes,
    select_text_blocks,
)


class ClassificationTests(unittest.TestCase):
    def test_classify_digital_page(self) -> None:
        signals = PageSignals(
            native_char_count=320,
            native_text_area_ratio=0.08,
            image_area_ratio=0.15,
            drawing_count=3,
        )
        self.assertEqual(classify_page(signals), "digital")

    def test_classify_scanned_page(self) -> None:
        signals = PageSignals(
            native_char_count=0,
            native_text_area_ratio=0.0,
            image_area_ratio=0.92,
            drawing_count=0,
        )
        self.assertEqual(classify_page(signals), "scanned")

    def test_classify_hybrid_page(self) -> None:
        signals = PageSignals(
            native_char_count=20,
            native_text_area_ratio=0.02,
            image_area_ratio=0.85,
            drawing_count=1,
        )
        self.assertEqual(classify_page(signals), "hybrid")


class BlockSelectionTests(unittest.TestCase):
    def test_hybrid_selection_filters_duplicate_ocr(self) -> None:
        native = [
            TextBlock(
                id="n1",
                source="native",
                bbox=(10, 10, 100, 30),
                text="Hello",
                confidence=0.99,
            )
        ]
        ocr = [
            TextBlock(
                id="o1",
                source="ocr",
                bbox=(12, 12, 98, 28),
                text="Hello",
                confidence=0.9,
            ),
            TextBlock(
                id="o2",
                source="ocr",
                bbox=(120, 40, 200, 80),
                text="World",
                confidence=0.9,
            ),
        ]
        selected = select_text_blocks("hybrid", native, ocr)
        self.assertEqual([block.id for block in selected], ["n1", "o2"])

    def test_intersection_ratio(self) -> None:
        ratio = intersection_ratio((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertAlmostEqual(ratio, 0.25)


class BackgroundModeTests(unittest.TestCase):
    def test_choose_elements_for_clean_digital_page(self) -> None:
        quality = QualityScore(
            text_confidence=0.98,
            layout_overlap_score=0.95,
            style_recovery_score=1.0,
            editable_ratio=1.0,
        )
        mode, reason = choose_background_mode(
            page_kind="digital",
            quality=quality,
            has_text=True,
            has_visuals=False,
        )
        self.assertEqual(mode, "elements")
        self.assertIsNone(reason)

    def test_choose_full_page_for_low_editable_score(self) -> None:
        quality = QualityScore(
            text_confidence=0.4,
            layout_overlap_score=0.3,
            style_recovery_score=0.2,
            editable_ratio=0.2,
        )
        mode, reason = choose_background_mode(
            page_kind="scanned",
            quality=quality,
            has_text=False,
            has_visuals=True,
        )
        self.assertEqual(mode, "full-page")
        self.assertIsNotNone(reason)

    def test_mask_text_regions_with_white_boxes(self) -> None:
        image = Image.new("RGB", (100, 60), color=(20, 30, 40))
        blocks = [
            TextBlock(
                id="ocr_1",
                source="ocr",
                bbox=(10, 10, 30, 25),
                text="demo",
                confidence=0.9,
                image_bbox=(10, 10, 30, 25),
            )
        ]
        masked = mask_text_regions_with_white_boxes(image, blocks, fitz.Rect(0, 0, 100, 60))
        self.assertEqual(masked.getpixel((0, 0)), (20, 30, 40))
        self.assertEqual(masked.getpixel((20, 16)), (255, 255, 255))

    def test_mask_prefers_image_polygon(self) -> None:
        image = Image.new("RGB", (60, 60), color=(10, 10, 10))
        blocks = [
            TextBlock(
                id="ocr_2",
                source="ocr",
                bbox=(0, 0, 60, 60),
                text="demo",
                confidence=0.9,
                image_polygon=((10, 10), (40, 10), (40, 20), (10, 20)),
            )
        ]
        masked = mask_text_regions_with_white_boxes(image, blocks, fitz.Rect(0, 0, 60, 60))
        self.assertEqual(masked.getpixel((15, 15)), (255, 255, 255))
        self.assertEqual(masked.getpixel((45, 25)), (10, 10, 10))

    def test_build_mask_shapes_uses_exact_polygon(self) -> None:
        blocks = [
            TextBlock(
                id="ocr_3",
                source="ocr",
                bbox=(0, 0, 60, 60),
                text="demo",
                confidence=0.9,
                image_polygon=((10, 10), (40, 10), (40, 20), (10, 20)),
            )
        ]
        shapes = build_mask_shapes(blocks, (60, 60), fitz.Rect(0, 0, 60, 60))
        self.assertEqual(shapes, [{"kind": "polygon", "points": [(10, 10), (40, 10), (40, 20), (10, 20)]}])


class CliTests(unittest.TestCase):
    def test_doc_unwarping_disabled_by_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.pdf", "output.pptx"])
        self.assertFalse(args.enable_doc_unwarping)

    def test_doc_unwarping_can_be_enabled(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.pdf", "output.pptx", "--enable-doc-unwarping"])
        self.assertTrue(args.enable_doc_unwarping)


class FontSizingTests(unittest.TestCase):
    def test_classify_text_script(self) -> None:
        self.assertEqual(classify_text_script("向量搜尋"), "cjk")
        self.assertEqual(classify_text_script("VECTOR"), "latin")
        self.assertEqual(classify_text_script("2026"), "numeric")
        self.assertEqual(classify_text_script("RAG 2026 向量"), "mixed")

    def test_estimate_font_size_grows_with_box_height(self) -> None:
        small = estimate_font_size("VECTOR", (0, 0, 120, 18))
        large = estimate_font_size("VECTOR", (0, 0, 120, 36))
        self.assertGreater(large, small)

    def test_estimate_font_size_stays_reasonable(self) -> None:
        estimate = estimate_font_size("RAG 2026：當向量搜尋遇上代理推理", (0, 0, 1200, 120))
        self.assertGreaterEqual(estimate, 6.0)
        self.assertLessEqual(estimate, 96.0)

    def test_build_text_fit_debug_entry_contains_error_metrics(self) -> None:
        block = TextBlock(
            id="ocr_debug_1",
            source="ocr",
            bbox=(0, 0, 200, 24),
            text="VECTOR EMBEDDINGS",
            confidence=0.9,
            font_size=18,
        )
        entry = build_text_fit_debug_entry(block)
        self.assertEqual(entry["id"], "ocr_debug_1")
        self.assertIn("width_error_ratio", entry)
        self.assertIn("height_error_ratio", entry)
        self.assertEqual(entry["target_bbox_pt"]["width"], 200.0)


if __name__ == "__main__":
    unittest.main()
