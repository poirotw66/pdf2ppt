from __future__ import annotations

import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
from PIL import Image, ImageDraw, ImageFont

from pdf2ppt.cli import build_parser, build_progress_callback, format_progress_line
from pdf2ppt.models import QualityScore, TextBlock
from pdf2ppt.pipeline import (
    BackgroundInpaintingError,
    build_text_fit_debug_entry,
    build_mask_shapes,
    build_text_mask_image,
    classify_text_script,
    ConversionOptions,
    default_font_family,
    estimate_font_size,
    estimate_background_complexity,
    estimate_text_bold,
    estimate_text_color,
    fit_text_frame,
    measure_text_dimensions,
    promote_ocr_bold_blocks,
    resolve_ocr_fit_max_size,
    PageSignals,
    choose_background_mode,
    classify_page,
    intersection_ratio,
    mask_text_regions_with_white_boxes,
    render_overlay_background,
    select_text_blocks,
    should_wrap_text_block,
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

    def test_build_text_mask_image_applies_padding(self) -> None:
        blocks = [
            TextBlock(
                id="ocr_4",
                source="ocr",
                bbox=(10, 10, 20, 20),
                text="demo",
                confidence=0.9,
                image_bbox=(10, 10, 20, 20),
            )
        ]
        mask = build_text_mask_image(blocks, (40, 40), fitz.Rect(0, 0, 40, 40), padding_px=2)
        self.assertEqual(mask.getpixel((10, 10)), 255)
        self.assertEqual(mask.getpixel((8, 10)), 255)
        self.assertEqual(mask.getpixel((5, 10)), 0)

    def test_render_overlay_background_auto_uses_opencv_fast_for_small_mask(self) -> None:
        image = Image.new("RGB", (60, 40), color=(35, 45, 55))
        blocks = [
            TextBlock(
                id="ocr_5",
                source="ocr",
                bbox=(20, 12, 35, 24),
                text="demo",
                confidence=0.9,
                image_bbox=(20, 12, 35, 24),
            )
        ]
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            inpaint_engine="auto",
            inpaint_padding_px=0,
            inpaint_max_area_ratio=0.2,
        )
        result = render_overlay_background(image, blocks, fitz.Rect(0, 0, 60, 40), options=options)
        self.assertEqual(result.engine_name, "opencv-fast")
        self.assertIn("opencv-fast", result.note or "")
        pixel = result.image.getpixel((25, 18))
        self.assertTrue(all(abs(channel - expected) <= 4 for channel, expected in zip(pixel, (35, 45, 55))))

    def test_render_overlay_background_auto_falls_back_to_white_box_for_large_mask(self) -> None:
        image = Image.new("RGB", (60, 40), color=(35, 45, 55))
        blocks = [
            TextBlock(
                id="ocr_6",
                source="ocr",
                bbox=(5, 5, 55, 35),
                text="demo",
                confidence=0.9,
                image_bbox=(5, 5, 55, 35),
            )
        ]
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            inpaint_engine="auto",
            inpaint_padding_px=0,
            inpaint_max_area_ratio=0.1,
        )
        result = render_overlay_background(image, blocks, fitz.Rect(0, 0, 60, 40), options=options)
        self.assertEqual(result.engine_name, "white-box")
        self.assertIn("white-box", result.note or "")
        self.assertEqual(result.image.getpixel((20, 18)), (255, 255, 255))

    def test_background_complexity_detects_textured_context(self) -> None:
        image = Image.linear_gradient("L").resize((80, 60)).convert("RGB")
        mask = Image.new("L", (80, 60), 0)
        for x in range(30, 50):
            for y in range(20, 35):
                mask.putpixel((x, y), 255)
        complexity = estimate_background_complexity(image, mask)
        self.assertGreater(complexity, 0.0)

    @patch("pdf2ppt.pipeline.shutil.which", return_value="/usr/bin/iopaint")
    @patch("pdf2ppt.pipeline.invoke_diffusion_backend")
    def test_render_overlay_background_uses_diffusion_when_explicit(
        self,
        invoke_backend: unittest.mock.Mock,
        _which: unittest.mock.Mock,
    ) -> None:
        image = Image.linear_gradient("L").resize((120, 90)).convert("RGB")
        invoke_backend.side_effect = (
            lambda crop_image, crop_mask, **kwargs: Image.new("RGB", crop_image.size, (90, 100, 110))
        )
        blocks = [
            TextBlock(
                id="ocr_7",
                source="ocr",
                bbox=(30, 20, 70, 40),
                text="demo",
                confidence=0.9,
                image_bbox=(30, 20, 70, 40),
            )
        ]
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            inpaint_engine="diffusion-local",
            inpaint_padding_px=4,
            diffusion_model="brushnet",
            diffusion_device="cuda",
            diffusion_max_crop_edge=256,
        )
        result = render_overlay_background(image, blocks, fitz.Rect(0, 0, 120, 90), options=options)
        self.assertEqual(result.engine_name, "diffusion-local")
        self.assertIn("brushnet", result.note or "")
        self.assertTrue(invoke_backend.called)

    @patch("pdf2ppt.pipeline.shutil.which", return_value="/usr/bin/iopaint")
    @patch("pdf2ppt.pipeline.invoke_diffusion_backend", side_effect=BackgroundInpaintingError("boom"))
    def test_render_overlay_background_diffusion_failure_falls_back_to_opencv(
        self,
        _invoke_backend: unittest.mock.Mock,
        _which: unittest.mock.Mock,
    ) -> None:
        image = Image.new("RGB", (80, 50), color=(20, 30, 40))
        blocks = [
            TextBlock(
                id="ocr_8",
                source="ocr",
                bbox=(25, 15, 40, 28),
                text="demo",
                confidence=0.9,
                image_bbox=(25, 15, 40, 28),
            )
        ]
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            inpaint_engine="diffusion-local",
            inpaint_padding_px=0,
            inpaint_max_area_ratio=0.5,
        )
        result = render_overlay_background(image, blocks, fitz.Rect(0, 0, 80, 50), options=options)
        self.assertEqual(result.engine_name, "opencv-fast")
        self.assertIn("Fallback to opencv-fast", result.note or "")


class CliTests(unittest.TestCase):
    def test_doc_unwarping_disabled_by_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.pdf", "output.pptx"])
        self.assertFalse(args.enable_doc_unwarping)
        self.assertEqual(args.inpaint_engine, "auto")
        self.assertIsNone(args.ocr_det_thresh)
        self.assertIsNone(args.ocr_det_box_thresh)
        self.assertIsNone(args.ocr_drop_score)
        self.assertEqual(args.inpaint_padding_px, 6)
        self.assertAlmostEqual(args.inpaint_max_area_ratio, 0.12)
        self.assertEqual(args.diffusion_command, "iopaint")
        self.assertEqual(args.diffusion_model, "brushnet")
        self.assertEqual(args.diffusion_device, "cuda")
        self.assertEqual(args.diffusion_max_crop_edge, 1024)
        self.assertAlmostEqual(args.diffusion_complexity_threshold, 0.3)

    def test_doc_unwarping_can_be_enabled(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.pdf", "output.pptx", "--enable-doc-unwarping"])
        self.assertTrue(args.enable_doc_unwarping)

    def test_inpaint_flags_can_be_configured(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "input.pdf",
                "output.pptx",
                "--ocr-det-thresh",
                "0.55",
                "--ocr-det-box-thresh",
                "0.6",
                "--ocr-drop-score",
                "0.65",
                "--inpaint-engine",
                "diffusion-local",
                "--inpaint-padding-px",
                "10",
                "--inpaint-max-area-ratio",
                "0.2",
                "--diffusion-command",
                "custom-iopaint",
                "--diffusion-model",
                "powerpaint-v2",
                "--diffusion-device",
                "cuda:0",
                "--diffusion-max-crop-edge",
                "768",
                "--diffusion-complexity-threshold",
                "0.45",
            ]
        )
        self.assertAlmostEqual(args.ocr_det_thresh, 0.55)
        self.assertAlmostEqual(args.ocr_det_box_thresh, 0.6)
        self.assertAlmostEqual(args.ocr_drop_score, 0.65)
        self.assertEqual(args.inpaint_engine, "diffusion-local")
        self.assertEqual(args.inpaint_padding_px, 10)
        self.assertAlmostEqual(args.inpaint_max_area_ratio, 0.2)
        self.assertEqual(args.diffusion_command, "custom-iopaint")
        self.assertEqual(args.diffusion_model, "powerpaint-v2")
        self.assertEqual(args.diffusion_device, "cuda:0")
        self.assertEqual(args.diffusion_max_crop_edge, 768)
        self.assertAlmostEqual(args.diffusion_complexity_threshold, 0.45)

    def test_format_progress_line_reports_completion(self) -> None:
        line = format_progress_line(2, 4, width=8)
        self.assertEqual(line, "Converting pages [####----] 2/4 ( 50%)")

    def test_build_progress_callback_writes_tty_progress_bar(self) -> None:
        class TtyBuffer(StringIO):
            def isatty(self) -> bool:
                return True

        buffer = TtyBuffer()
        callback = build_progress_callback(buffer, width=4)
        callback(0, 2)
        callback(1, 2)
        callback(2, 2)
        self.assertIn("\rConverting pages [----] 0/2 (  0%)", buffer.getvalue())
        self.assertIn("\rConverting pages [##--] 1/2 ( 50%)", buffer.getvalue())
        self.assertTrue(buffer.getvalue().endswith("\rConverting pages [####] 2/2 (100%)\n"))


class FontSizingTests(unittest.TestCase):
    def test_default_font_family_by_script(self) -> None:
        self.assertEqual(default_font_family("latin"), "DejaVu Sans")
        self.assertEqual(default_font_family("cjk"), "Noto Sans CJK TC")

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

    def test_fit_text_frame_uses_fit_text_for_ocr_blocks(self) -> None:
        calls: list[dict[str, object]] = []

        class DummyTextFrame:
            def fit_text(self, **kwargs: object) -> None:
                calls.append(kwargs)

        block = TextBlock(
            id="ocr_fit_1",
            source="ocr",
            bbox=(0, 0, 140, 24),
            text="VECTOR EMBEDDINGS",
            confidence=0.9,
            font_size=12,
        )
        self.assertTrue(fit_text_frame(DummyTextFrame(), block, scale_x=1.0, scale_y=1.0))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["font_family"], "DejaVu Sans")
        self.assertLessEqual(calls[0]["max_size"], 12)
        self.assertGreaterEqual(calls[0]["max_size"], 6)

    def test_should_wrap_text_block_disables_wrap_for_single_line_ocr(self) -> None:
        self.assertFalse(
            should_wrap_text_block(
                TextBlock(
                    id="ocr_wrap_1",
                    source="ocr",
                    bbox=(0, 0, 100, 20),
                    text="VECTOR EMBEDDINGS",
                    confidence=0.9,
                )
            )
        )
        self.assertTrue(
            should_wrap_text_block(
                TextBlock(
                    id="ocr_wrap_2",
                    source="ocr",
                    bbox=(0, 0, 100, 20),
                    text="LINE 1\nLINE 2",
                    confidence=0.9,
                )
            )
        )

    def test_resolve_ocr_fit_max_size_respects_single_line_width(self) -> None:
        size = resolve_ocr_fit_max_size(
            TextBlock(
                id="ocr_width_1",
                source="ocr",
                bbox=(0, 0, 120, 24),
                text="VECTOR EMBEDDINGS",
                confidence=0.9,
                font_size=12,
            ),
            font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            base_size=12,
            scale_x=1.0,
            scale_y=1.0,
            script="latin",
        )
        measured_width, _ = measure_text_dimensions(
            "VECTOR EMBEDDINGS",
            size,
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        self.assertLessEqual(measured_width, 120 * 0.96)


class TextColorTests(unittest.TestCase):
    def test_estimate_text_color_prefers_dark_foreground(self) -> None:
        color = Image.new("RGB", (20, 12), (240, 242, 245))
        gray = color.convert("L")
        for x in range(4, 16):
            for y in range(3, 9):
                color.putpixel((x, y), (32, 92, 184))
                gray.putpixel((x, y), 70)
        self.assertEqual(estimate_text_color(color, gray), "#205CB8")

    def test_estimate_text_color_prefers_light_foreground(self) -> None:
        color = Image.new("RGB", (20, 12), (24, 28, 36))
        gray = color.convert("L")
        for x in range(4, 16):
            for y in range(3, 9):
                color.putpixel((x, y), (242, 244, 248))
                gray.putpixel((x, y), 240)
        self.assertEqual(estimate_text_color(color, gray), "#F2F4F8")


class TextStyleTests(unittest.TestCase):
    def test_estimate_text_bold_distinguishes_regular_and_bold(self) -> None:
        regular_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
        bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)

        def render(font: ImageFont.FreeTypeFont) -> Image.Image:
            image = Image.new("L", (220, 70), 255)
            draw = ImageDraw.Draw(image)
            draw.text((10, 12), "VECTOR", font=font, fill=0)
            return image

        regular = render(regular_font)
        bold = render(bold_font)
        self.assertFalse(estimate_text_bold("VECTOR", regular))
        self.assertTrue(estimate_text_bold("VECTOR", bold))

    def test_promote_ocr_bold_blocks_marks_titles_bold(self) -> None:
        blocks = [
            TextBlock(
                id="ocr_title_1",
                source="ocr",
                bbox=(0, 0, 300, 80),
                text="Main Title",
                confidence=0.9,
                font_size=42,
                block_role="title",
                bold=False,
            ),
            TextBlock(
                id="ocr_body_1",
                source="ocr",
                bbox=(0, 0, 200, 30),
                text="body text",
                confidence=0.9,
                font_size=18,
                block_role="body",
                bold=False,
            ),
        ]
        promote_ocr_bold_blocks(blocks)
        self.assertTrue(blocks[0].bold)
        self.assertFalse(blocks[1].bold)

    def test_promote_ocr_bold_blocks_does_not_promote_body_headers(self) -> None:
        blocks = [
            TextBlock(
                id="ocr_header_1",
                source="ocr",
                bbox=(0, 0, 220, 40),
                text="核心優勢",
                confidence=0.9,
                font_size=29.5,
                block_role="body",
                bold=False,
            ),
            TextBlock(
                id="ocr_body_2",
                source="ocr",
                bbox=(0, 0, 300, 35),
                text="這是一段較長的正文內容",
                confidence=0.9,
                font_size=27.0,
                block_role="body",
                bold=False,
            ),
            TextBlock(
                id="ocr_body_3",
                source="ocr",
                bbox=(0, 0, 300, 35),
                text="普通內文",
                confidence=0.9,
                font_size=18.0,
                block_role="body",
                bold=False,
            ),
        ]
        promote_ocr_bold_blocks(blocks)
        self.assertFalse(blocks[0].bold)
        self.assertFalse(blocks[1].bold)
        self.assertFalse(blocks[2].bold)


if __name__ == "__main__":
    unittest.main()
