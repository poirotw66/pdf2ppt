"""Per-page timeout helper and OCR/layout timeout fallbacks (mocked)."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from pdf2ppt.core import ConversionOptions, OcrPageData
from pdf2ppt.pipeline import analyze_page, prepare_batched_ocr_page_data
from pdf2ppt.timeouts import call_with_timeout


class CallWithTimeoutTests(unittest.TestCase):
    def test_returns_result_when_fast(self) -> None:
        self.assertEqual(call_with_timeout(lambda: 42, 1.0, label="fast"), 42)

    def test_disabled_timeout_runs_inline(self) -> None:
        calls = {"n": 0}

        def work() -> str:
            calls["n"] += 1
            return "ok"

        self.assertEqual(call_with_timeout(work, 0, label="inline"), "ok")
        self.assertEqual(call_with_timeout(work, None, label="inline"), "ok")
        self.assertEqual(calls["n"], 2)

    def test_raises_timeout_for_slow_function(self) -> None:
        started = threading.Event()

        def slow() -> None:
            started.set()
            time.sleep(5.0)

        with self.assertRaises(TimeoutError) as ctx:
            call_with_timeout(slow, 0.15, label="slow-op")
        self.assertIn("slow-op", str(ctx.exception))
        self.assertTrue(started.wait(1.0))


class PageOcrTimeoutFallbackTests(unittest.TestCase):
    def test_analyze_page_ocr_timeout_uses_empty_ocr(self) -> None:
        import fitz

        document = fitz.open()
        page = document.new_page(width=200, height=200)
        # No native text → OCR path.
        options = ConversionOptions(
            input_path=MagicMock(),
            output_path=MagicMock(),
            report_path=MagicMock(),
            page_timeout_s=0.2,
            layout_engine="heuristic",
            vector_mode="off",
            inpaint_engine="opencv-fast",
        )
        ocr_engine = MagicMock()

        def slow_ocr(_image: Image.Image, _page_number: int) -> OcrPageData:
            time.sleep(5.0)
            return OcrPageData(blocks=[], image=Image.new("RGB", (10, 10)))

        ocr_engine.extract_text_blocks.side_effect = slow_ocr

        with patch("pdf2ppt.pipeline.render_page_image", return_value=Image.new("RGB", (20, 20))):
            result = analyze_page(page, options, ocr_engine)

        self.assertEqual(result.text_blocks, [])
        # Scanned/empty OCR → full-page raster fallback rather than hanging.
        self.assertEqual(result.background_mode, "full-page")
        document.close()

    def test_batched_ocr_timeout_yields_empty_pages(self) -> None:
        import fitz

        document = fitz.open()
        document.new_page(width=100, height=100)
        options = ConversionOptions(
            input_path=MagicMock(),
            output_path=MagicMock(),
            report_path=MagicMock(),
            page_timeout_s=0.2,
            ocr_batch_size=1,
            vector_mode="off",
        )
        ocr_engine = MagicMock()

        def slow_batch(images: list, numbers: list) -> list[OcrPageData]:
            time.sleep(5.0)
            return [OcrPageData(blocks=[], image=img) for img in images]

        ocr_engine.extract_text_blocks_batch.side_effect = slow_batch

        with patch("pdf2ppt.pipeline.extract_native_text_blocks", return_value=([], [])):
            with patch(
                "pdf2ppt.pipeline.compute_page_signals",
                return_value=MagicMock(
                    native_char_count=0,
                    native_text_area_ratio=0.0,
                    image_area_ratio=1.0,
                    drawing_count=0,
                ),
            ):
                with patch("pdf2ppt.pipeline.classify_page", return_value="scanned"):
                    with patch(
                        "pdf2ppt.pipeline.render_page_image",
                        return_value=Image.new("RGB", (8, 8)),
                    ):
                        results = prepare_batched_ocr_page_data(document, options, ocr_engine)

        self.assertEqual(len(results), 1)
        page_image, ocr_data = results[1]
        self.assertEqual(ocr_data.blocks, [])
        self.assertIs(ocr_data.image, page_image)
        document.close()


if __name__ == "__main__":
    unittest.main()
