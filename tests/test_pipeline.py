from __future__ import annotations

import subprocess
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pdf2ppt.cli import build_parser, build_progress_callback, format_progress_line, main
from pdf2ppt.inpainting_overlay import _apply_targeted_file_back_color_correction
from pdf2ppt.models import QualityScore, TextBlock
from pdf2ppt.ocr import build_local_ocr_model_kwargs, ensure_local_model_dir
from pdf2ppt.pipeline import (
    BackgroundInpaintingError,
    analyze_page,
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
    filter_suspicious_ocr_blocks,
    invoke_diffusion_backend,
    measure_text_dimensions,
    OpenCvFastInpaintingEngine,
    pil_to_image_bytes,
    promote_ocr_bold_blocks,
    resolve_background_render_dpi,
    resolve_ocr_fit_max_size,
    resolve_vertical_anchor,
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

    def test_filter_suspicious_ocr_blocks_drops_large_low_confidence_short_text(self) -> None:
        blocks = [
            TextBlock(
                id="ocr_bad",
                source="ocr",
                bbox=(600, 350, 780, 590),
                text="具",
                confidence=0.15,
            ),
            TextBlock(
                id="ocr_good",
                source="ocr",
                bbox=(200, 360, 520, 400),
                text="縮小至50-100份候選文檔",
                confidence=0.94,
            ),
        ]
        filtered = filter_suspicious_ocr_blocks(blocks, fitz.Rect(0, 0, 1376, 768))
        self.assertEqual([block.id for block in filtered], ["ocr_good"])

    def test_filter_suspicious_ocr_blocks_keeps_small_low_confidence_short_text(self) -> None:
        blocks = [
            TextBlock(
                id="ocr_small",
                source="ocr",
                bbox=(100, 100, 130, 124),
                text="A",
                confidence=0.2,
            )
        ]
        filtered = filter_suspicious_ocr_blocks(blocks, fitz.Rect(0, 0, 1376, 768))
        self.assertEqual([block.id for block in filtered], ["ocr_small"])


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

    def test_build_mask_shapes_scales_page_space_polygon_to_target_image(self) -> None:
        blocks = [
            TextBlock(
                id="ocr_3b",
                source="ocr",
                bbox=(0, 0, 60, 60),
                text="demo",
                confidence=0.9,
                image_polygon=((20, 20), (80, 20), (80, 40), (20, 40)),
            )
        ]
        shapes = build_mask_shapes(blocks, (50, 50), fitz.Rect(0, 0, 100, 100))
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
        grid_y, grid_x = np.indices((40, 60), dtype=np.uint8)
        textured = np.stack(
            [
                (grid_x * 17 + grid_y * 11) % 255,
                (grid_x * 29 + grid_y * 7) % 255,
                (grid_x * 13 + grid_y * 19) % 255,
            ],
            axis=2,
        )
        image = Image.fromarray(textured, mode="RGB")
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

    def test_render_overlay_background_auto_uses_opencv_fast_for_large_low_texture_mask(self) -> None:
        width, height = 180, 120
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        base = np.stack(
            [
                150.0 + 0.03 * grid_x + 0.08 * grid_y,
                140.0 + 0.02 * grid_x + 0.06 * grid_y,
                130.0 + 0.01 * grid_x + 0.05 * grid_y,
            ],
            axis=2,
        ).clip(0, 255).astype("uint8")
        image = Image.fromarray(base, mode="RGB")
        blocks = [
            TextBlock(
                id="ocr_large_flat",
                source="ocr",
                bbox=(45, 25, 135, 95),
                text="demo",
                confidence=0.9,
                image_bbox=(45, 25, 135, 95),
            )
        ]
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            inpaint_engine="auto",
            inpaint_padding_px=0,
            inpaint_max_area_ratio=0.12,
        )
        result = render_overlay_background(image, blocks, fitz.Rect(0, 0, width, height), options=options)
        self.assertEqual(result.engine_name, "opencv-fast")
        self.assertIn("low-texture", result.note or "")

        result_array = np.array(result.image, dtype=np.int16)
        base_array = base.astype(np.int16)
        mask = np.zeros((height, width), dtype=bool)
        mask[25:95, 45:135] = True
        mae = np.abs(result_array - base_array)[mask].mean()
        self.assertLess(mae, 1.0)

    def test_background_complexity_detects_textured_context(self) -> None:
        image = Image.linear_gradient("L").resize((80, 60)).convert("RGB")
        mask = Image.new("L", (80, 60), 0)
        for x in range(30, 50):
            for y in range(20, 35):
                mask.putpixel((x, y), 255)
        complexity = estimate_background_complexity(image, mask)
        self.assertGreater(complexity, 0.0)

    def test_opencv_fast_restores_smooth_gradient_background(self) -> None:
        width, height = 180, 120
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        base = np.stack(
            [
                150.0 + 0.03 * grid_x + 0.08 * grid_y,
                140.0 + 0.02 * grid_x + 0.06 * grid_y,
                130.0 + 0.01 * grid_x + 0.05 * grid_y,
            ],
            axis=2,
        ).clip(0, 255).astype("uint8")
        mask = np.zeros((height, width), dtype="uint8")
        mask[25:95, 45:135] = 255

        result = OpenCvFastInpaintingEngine().inpaint(
            Image.fromarray(base, mode="RGB"),
            Image.fromarray(mask, mode="L"),
        )

        result_array = np.array(result, dtype=np.int16)
        base_array = base.astype(np.int16)
        mae = np.abs(result_array - base_array)[mask > 0].mean()
        self.assertLess(mae, 1.0)

    def test_opencv_fast_cleans_tight_mask_text_halo_on_smooth_background(self) -> None:
        width, height = 320, 160
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        base = np.stack(
            [
                166.0 + 0.02 * grid_x + 0.03 * grid_y,
                118.0 + 0.01 * grid_x + 0.02 * grid_y,
                182.0 + 0.015 * grid_x + 0.015 * grid_y,
            ],
            axis=2,
        ).clip(0, 255).astype("uint8")
        image = Image.fromarray(base, mode="RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        text = "File-Back"
        bbox = draw.textbbox((48, 52), text, font=font)
        draw.text((48, 52), text, fill=(247, 237, 248), font=font)

        mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle((bbox[0] + 2, bbox[1] + 2, bbox[2] - 2, bbox[3] - 2), fill=255)

        result = OpenCvFastInpaintingEngine().inpaint(image, mask)

        x0 = max(0, bbox[0] - 3)
        y0 = max(0, bbox[1] - 3)
        x1 = min(width, bbox[2] + 3)
        y1 = min(height, bbox[3] + 3)
        result_array = np.array(result, dtype=np.int16)
        base_array = base.astype(np.int16)
        mae = np.abs(result_array[y0:y1, x0:x1] - base_array[y0:y1, x0:x1]).mean()
        self.assertLess(mae, 1.0)

    def test_opencv_fast_restores_curved_gradient_background(self) -> None:
        width, height = 260, 160
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        channel_r = 90.0 + 0.55 * grid_x + 0.18 * grid_y + 0.0022 * ((grid_x - 140.0) ** 2)
        channel_r += 0.0008 * ((grid_y - 80.0) ** 2)
        channel_g = 70.0 + 0.32 * grid_x + 0.24 * grid_y + 0.0015 * ((grid_x - 120.0) ** 2)
        channel_b = 120.0 + 0.18 * grid_x + 0.16 * grid_y + 0.0018 * ((grid_x - 150.0) ** 2)
        channel_b += 0.0009 * grid_x * grid_y / 10.0
        base = np.stack(
            [channel_r, channel_g, channel_b],
            axis=2,
        ).clip(0, 255).astype("uint8")
        mask = np.zeros((height, width), dtype="uint8")
        mask[48:118, 70:210] = 255

        result = OpenCvFastInpaintingEngine().inpaint(
            Image.fromarray(base, mode="RGB"),
            Image.fromarray(mask, mode="L"),
        )

        result_array = np.array(result, dtype=np.int16)
        base_array = base.astype(np.int16)
        mae = np.abs(result_array - base_array)[mask > 0].mean()
        self.assertLess(mae, 1.5)

    def test_targeted_file_back_lab_correction_improves_local_ring_alignment(self) -> None:
        width, height = 260, 160
        grid_y, grid_x = np.indices((height, width), dtype=np.float32)
        base = np.stack(
            [
                195.0 + 0.14 * grid_x + 0.08 * grid_y,
                170.0 + 0.10 * grid_x + 0.07 * grid_y,
                205.0 + 0.12 * grid_x + 0.05 * grid_y,
            ],
            axis=2,
        ).clip(0, 255).astype("uint8")
        page_image = Image.fromarray(base, mode="RGB")
        repaired = base.astype(np.float32)
        repaired_region = repaired[52:108, 78:198]
        repaired_region_mean = repaired_region.mean(axis=(0, 1), keepdims=True)
        repaired[52:108, 78:198] = repaired_region_mean + (repaired_region - repaired_region_mean) * 0.45
        repaired[52:108, 78:198, 0] -= 20.0
        repaired[52:108, 78:198, 1] -= 14.0
        repaired[52:108, 78:198, 2] -= 18.0
        repaired = np.clip(repaired, 0, 255).astype("uint8")
        repaired_image = Image.fromarray(repaired, mode="RGB")
        block = TextBlock(
            id="ocr_file_back",
            source="ocr",
            bbox=(78, 52, 198, 108),
            text="歸檔(File-Back)",
            confidence=0.95,
            image_bbox=(78, 52, 198, 108),
        )
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            inpaint_engine="opencv-fast",
            inpaint_padding_px=0,
        )

        corrected_image, debug_images, note = _apply_targeted_file_back_color_correction(
            page_image,
            repaired_image,
            [block],
            fitz.Rect(0, 0, width, height),
            options=options,
        )

        repaired_arr = np.array(repaired_image, dtype=np.uint8)
        corrected_arr = np.array(corrected_image, dtype=np.uint8)
        component = np.zeros((height, width), dtype=np.uint8)
        component[52:108, 78:198] = 1
        inner = cv2.dilate(component, np.ones((13, 13), np.uint8), iterations=1).astype(bool)
        outer = cv2.dilate(inner.astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1).astype(bool)
        ring = outer & (~inner)
        region = component.astype(bool)
        ring_mean = base[ring].astype(np.float32).mean(axis=0)
        repaired_gap = np.abs(repaired_arr[region].astype(np.float32).mean(axis=0) - ring_mean).mean()
        corrected_gap = np.abs(corrected_arr[region].astype(np.float32).mean(axis=0) - ring_mean).mean()
        repaired_lab = cv2.cvtColor(repaired_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        corrected_lab = cv2.cvtColor(corrected_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        repaired_span = np.percentile(repaired_lab[:, :, 0][region], 90) - np.percentile(
            repaired_lab[:, :, 0][region], 10
        )
        corrected_span = np.percentile(corrected_lab[:, :, 0][region], 90) - np.percentile(
            corrected_lab[:, :, 0][region], 10
        )

        self.assertLess(corrected_gap, repaired_gap)
        self.assertGreater(corrected_span, repaired_span)
        self.assertIn("File-Back", note or "")
        self.assertIn("file_back_corrected", debug_images)
        self.assertIn("file_back_mask", debug_images)

    @patch("pdf2ppt.inpainting_engines.shutil.which", return_value="/usr/bin/iopaint")
    @patch("pdf2ppt.inpainting_engines.invoke_diffusion_backend")
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

    @patch("pdf2ppt.inpainting_engines.shutil.which", return_value="/usr/bin/iopaint")
    @patch("pdf2ppt.inpainting_engines.invoke_diffusion_backend", side_effect=BackgroundInpaintingError("boom"))
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

    @patch("pdf2ppt.inpainting_engines.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="iopaint", timeout=2.5))
    def test_invoke_diffusion_backend_timeout_raises_background_error(
        self,
        _run: unittest.mock.Mock,
    ) -> None:
        with self.assertRaises(BackgroundInpaintingError) as context:
            invoke_diffusion_backend(
                Image.new("RGB", (20, 20), color=(0, 0, 0)),
                Image.new("L", (20, 20), color=255),
                command="iopaint",
                model="brushnet",
                device="cpu",
                max_crop_edge=64,
                timeout_sec=2.5,
            )
        self.assertIn("timed out", str(context.exception))


class AnalyzePagePerformanceTests(unittest.TestCase):
    @patch("pdf2ppt.pipeline.extract_image_elements", return_value=[])
    @patch("pdf2ppt.pipeline.render_page_image")
    @patch("pdf2ppt.pipeline.choose_background_mode", return_value=("elements", None))
    @patch(
        "pdf2ppt.pipeline.score_page",
        return_value=QualityScore(
            text_confidence=0.98,
            layout_overlap_score=0.95,
            style_recovery_score=1.0,
            editable_ratio=1.0,
        ),
    )
    @patch("pdf2ppt.pipeline.select_text_blocks")
    @patch("pdf2ppt.pipeline.classify_page", return_value="digital")
    @patch(
        "pdf2ppt.pipeline.compute_page_signals",
        return_value=PageSignals(
            native_char_count=100,
            native_text_area_ratio=0.2,
            image_area_ratio=0.0,
            drawing_count=0,
        ),
    )
    @patch("pdf2ppt.pipeline.extract_native_text_blocks")
    def test_analyze_page_skips_render_for_elements_mode(
        self,
        extract_native_text_blocks_mock: unittest.mock.Mock,
        _compute_page_signals_mock: unittest.mock.Mock,
        _classify_page_mock: unittest.mock.Mock,
        select_text_blocks_mock: unittest.mock.Mock,
        _score_page_mock: unittest.mock.Mock,
        _choose_background_mode_mock: unittest.mock.Mock,
        render_page_image_mock: unittest.mock.Mock,
        extract_image_elements_mock: unittest.mock.Mock,
    ) -> None:
        native_blocks = [
            TextBlock(
                id="n1",
                source="native",
                bbox=(10, 10, 100, 30),
                text="Hello",
                confidence=0.99,
            )
        ]
        extract_native_text_blocks_mock.return_value = (native_blocks, [])
        select_text_blocks_mock.return_value = native_blocks
        page = SimpleNamespace(number=0, rect=fitz.Rect(0, 0, 320, 240))
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
        )

        result = analyze_page(page, options, ocr_engine=unittest.mock.Mock())

        self.assertEqual(result.background_mode, "elements")
        render_page_image_mock.assert_not_called()
        extract_image_elements_mock.assert_called_once()

    @patch("pdf2ppt.pipeline.pil_to_image_bytes", return_value=b"jpeg-bytes")
    @patch("pdf2ppt.pipeline.render_page_image")
    @patch("pdf2ppt.pipeline.choose_background_mode", return_value=("full-page", "fallback"))
    @patch(
        "pdf2ppt.pipeline.score_page",
        return_value=QualityScore(
            text_confidence=0.8,
            layout_overlap_score=0.7,
            style_recovery_score=0.45,
            editable_ratio=0.2,
        ),
    )
    @patch("pdf2ppt.pipeline.select_text_blocks", return_value=[])
    @patch("pdf2ppt.pipeline.classify_page", return_value="scanned")
    @patch(
        "pdf2ppt.pipeline.compute_page_signals",
        return_value=PageSignals(
            native_char_count=0,
            native_text_area_ratio=0.0,
            image_area_ratio=1.0,
            drawing_count=0,
        ),
    )
    @patch("pdf2ppt.pipeline.extract_native_text_blocks", return_value=([], []))
    def test_analyze_page_renders_background_with_background_dpi(
        self,
        _extract_native_text_blocks_mock: unittest.mock.Mock,
        _compute_page_signals_mock: unittest.mock.Mock,
        _classify_page_mock: unittest.mock.Mock,
        _select_text_blocks_mock: unittest.mock.Mock,
        _score_page_mock: unittest.mock.Mock,
        _choose_background_mode_mock: unittest.mock.Mock,
        render_page_image_mock: unittest.mock.Mock,
        encode_background_mock: unittest.mock.Mock,
    ) -> None:
        ocr_image = Image.new("RGB", (200, 100), (10, 10, 10))
        render_page_image_mock.return_value = ocr_image
        page = SimpleNamespace(number=0, rect=fitz.Rect(0, 0, 320, 240))
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            render_dpi=144,
            background_dpi=96,
            background_image_format="jpeg",
            background_jpeg_quality=70,
        )
        ocr_engine = unittest.mock.Mock()
        ocr_engine.extract_text_blocks.return_value = SimpleNamespace(blocks=[], image=ocr_image)

        result = analyze_page(page, options, ocr_engine=ocr_engine)

        self.assertEqual(result.background_image_bytes, b"jpeg-bytes")
        self.assertEqual(render_page_image_mock.call_args_list[0].kwargs["dpi"], 144)
        self.assertEqual(render_page_image_mock.call_count, 1)
        encode_background_mock.assert_called_once_with(
            unittest.mock.ANY,
            image_format="jpeg",
            jpeg_quality=70,
        )
        resized_background = encode_background_mock.call_args.args[0]
        self.assertEqual(resized_background.size, (133, 67))

    @patch("pdf2ppt.pipeline.pil_to_image_bytes", return_value=b"jpeg-bytes")
    @patch("pdf2ppt.pipeline.render_page_image")
    @patch("pdf2ppt.pipeline.choose_background_mode", return_value=("full-page", "fallback"))
    @patch(
        "pdf2ppt.pipeline.score_page",
        return_value=QualityScore(
            text_confidence=0.8,
            layout_overlap_score=0.7,
            style_recovery_score=0.45,
            editable_ratio=0.2,
        ),
    )
    @patch("pdf2ppt.pipeline.select_text_blocks", return_value=[])
    @patch("pdf2ppt.pipeline.classify_page", return_value="scanned")
    @patch(
        "pdf2ppt.pipeline.compute_page_signals",
        return_value=PageSignals(
            native_char_count=0,
            native_text_area_ratio=0.0,
            image_area_ratio=1.0,
            drawing_count=0,
        ),
    )
    @patch("pdf2ppt.pipeline.extract_native_text_blocks", return_value=([], []))
    def test_analyze_page_rerenders_background_when_doc_orientation_enabled(
        self,
        _extract_native_text_blocks_mock: unittest.mock.Mock,
        _compute_page_signals_mock: unittest.mock.Mock,
        _classify_page_mock: unittest.mock.Mock,
        _select_text_blocks_mock: unittest.mock.Mock,
        _score_page_mock: unittest.mock.Mock,
        _choose_background_mode_mock: unittest.mock.Mock,
        render_page_image_mock: unittest.mock.Mock,
        encode_background_mock: unittest.mock.Mock,
    ) -> None:
        ocr_image = Image.new("RGB", (200, 100), (10, 10, 10))
        background_image = Image.new("RGB", (120, 60), (20, 20, 20))
        render_page_image_mock.side_effect = [ocr_image, background_image]
        page = SimpleNamespace(number=0, rect=fitz.Rect(0, 0, 320, 240))
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            render_dpi=144,
            background_dpi=96,
            background_image_format="jpeg",
            background_jpeg_quality=70,
            ocr_use_doc_orientation=True,
        )
        ocr_engine = unittest.mock.Mock()
        ocr_engine.extract_text_blocks.return_value = SimpleNamespace(blocks=[], image=ocr_image)

        result = analyze_page(page, options, ocr_engine=ocr_engine)

        self.assertEqual(result.background_image_bytes, b"jpeg-bytes")
        self.assertEqual(render_page_image_mock.call_args_list[0].kwargs["dpi"], 144)
        self.assertEqual(render_page_image_mock.call_args_list[1].kwargs["dpi"], 96)
        encode_background_mock.assert_called_once_with(
            background_image,
            image_format="jpeg",
            jpeg_quality=70,
        )


class OcrModelConfigTests(unittest.TestCase):
    def test_build_local_ocr_model_kwargs_uses_repo_local_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_root = Path(tmp_dir) / "model"
            kwargs = build_local_ocr_model_kwargs(
                model_root=model_root,
                lang="ch",
                use_doc_orientation=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            self.assertEqual(kwargs["text_detection_model_name"], "PP-OCRv5_server_det")
            self.assertEqual(kwargs["text_recognition_model_name"], "PP-OCRv5_server_rec")
            self.assertEqual(Path(kwargs["text_detection_model_dir"]).parent, model_root.resolve())
            self.assertNotIn("doc_orientation_classify_model_dir", kwargs)
            self.assertNotIn("textline_orientation_model_dir", kwargs)

    def test_build_local_ocr_model_kwargs_can_enable_doc_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_root = Path(tmp_dir) / "model"
            kwargs = build_local_ocr_model_kwargs(
                model_root=model_root,
                lang="ch",
                use_doc_orientation=True,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            self.assertEqual(kwargs["doc_orientation_classify_model_name"], "PP-LCNet_x1_0_doc_ori")
            self.assertEqual(Path(kwargs["doc_orientation_classify_model_dir"]).parent, model_root.resolve())

    def test_ensure_local_model_dir_reuses_existing_paddlex_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = Path(tmp_dir)
            cache_root = temp_root / "cache"
            cached_model_dir = cache_root / "PP-OCRv5_server_det"
            cached_model_dir.mkdir(parents=True)
            (cached_model_dir / "inference.json").write_text("{}", encoding="utf-8")

            with patch("pdf2ppt.ocr.PADDLEX_OFFICIAL_MODEL_CACHE_DIR", cache_root):
                local_model_dir = ensure_local_model_dir(temp_root / "model", "PP-OCRv5_server_det")

            self.assertTrue(local_model_dir.exists())
            self.assertTrue((local_model_dir / "inference.json").exists())


class CliTests(unittest.TestCase):
    def test_doc_unwarping_disabled_by_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.pdf", "output.pptx"])
        self.assertFalse(args.enable_doc_unwarping)
        self.assertFalse(args.enable_doc_orientation)
        self.assertEqual(args.ocr_model_root, Path("model"))
        self.assertFalse(args.enable_textline_orientation)
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
        self.assertAlmostEqual(args.diffusion_timeout_sec, 120.0)
        self.assertEqual(args.background_dpi, 110)
        self.assertEqual(args.background_format, "jpeg")
        self.assertEqual(args.background_jpeg_quality, 82)
        self.assertEqual(args.log_level, "INFO")

    def test_doc_unwarping_can_be_enabled(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.pdf", "output.pptx", "--enable-doc-unwarping"])
        self.assertTrue(args.enable_doc_unwarping)

    def test_orientation_flags_and_model_root_can_be_enabled(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "input.pdf",
                "output.pptx",
                "--ocr-model-root",
                "local-models",
                "--enable-doc-orientation",
                "--enable-textline-orientation",
            ]
        )
        self.assertEqual(args.ocr_model_root, Path("local-models"))
        self.assertTrue(args.enable_doc_orientation)
        self.assertTrue(args.enable_textline_orientation)

    @patch("pdf2ppt.cli.convert_pdf")
    def test_main_does_not_enable_debug_dir_by_default(self, convert_pdf_mock: unittest.mock.Mock) -> None:
        convert_pdf_mock.return_value = SimpleNamespace(
            input_path="input.pdf",
            output_path="output.pptx",
            pages=[object()],
        )

        exit_code = main(["input.pdf", "output.pptx"])

        self.assertEqual(exit_code, 0)
        options = convert_pdf_mock.call_args.args[0]
        self.assertIsNone(options.debug_dir)

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
                "--diffusion-timeout-sec",
                "15",
                "--background-dpi",
                "96",
                "--background-format",
                "png",
                "--background-jpeg-quality",
                "70",
                "--log-level",
                "DEBUG",
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
        self.assertAlmostEqual(args.diffusion_timeout_sec, 15.0)
        self.assertEqual(args.background_dpi, 96)
        self.assertEqual(args.background_format, "png")
        self.assertEqual(args.background_jpeg_quality, 70)
        self.assertEqual(args.log_level, "DEBUG")

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
    def test_pil_to_image_bytes_supports_jpeg(self) -> None:
        image = Image.new("RGB", (24, 24), (120, 140, 160))
        encoded = pil_to_image_bytes(image, image_format="jpeg", jpeg_quality=70)
        self.assertTrue(encoded.startswith(b"\xff\xd8\xff"))

    def test_resolve_background_render_dpi_uses_option(self) -> None:
        options = ConversionOptions(
            input_path=Path("input.pdf"),
            output_path=Path("output.pptx"),
            report_path=Path("output.report.json"),
            background_dpi=96,
        )
        self.assertEqual(resolve_background_render_dpi(options), 96)

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
        self.assertGreaterEqual(calls[0]["max_size"], 6)

    @patch("pdf2ppt.ppt_render.resolve_ocr_fit_max_size", return_value=18)
    def test_fit_text_frame_uses_resolved_cap_for_ocr_blocks(self, _resolve_max_size: unittest.mock.Mock) -> None:
        calls: list[dict[str, object]] = []

        class DummyTextFrame:
            def fit_text(self, **kwargs: object) -> None:
                calls.append(kwargs)

        block = TextBlock(
            id="ocr_fit_2",
            source="ocr",
            bbox=(0, 0, 200, 36),
            text="NotebookLM",
            confidence=0.9,
            font_size=12,
        )
        self.assertTrue(fit_text_frame(DummyTextFrame(), block, scale_x=1.0, scale_y=1.0))
        self.assertEqual(calls[0]["max_size"], 18)

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

    def test_resolve_vertical_anchor_uses_middle_for_single_line_ocr(self) -> None:
        anchor = resolve_vertical_anchor(
            TextBlock(
                id="ocr_anchor_1",
                source="ocr",
                bbox=(0, 0, 100, 20),
                text="VECTOR EMBEDDINGS",
                confidence=0.9,
            )
        )
        self.assertEqual(int(anchor), 3)

    def test_resolve_vertical_anchor_uses_top_for_multiline_text(self) -> None:
        anchor = resolve_vertical_anchor(
            TextBlock(
                id="ocr_anchor_2",
                source="ocr",
                bbox=(0, 0, 100, 40),
                text="LINE 1\nLINE 2",
                confidence=0.9,
            )
        )
        self.assertEqual(int(anchor), 1)

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
