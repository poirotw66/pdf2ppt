"""Per-page timeout helper and OCR/layout timeout fallbacks (mocked)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from pdf2ppt.core import ConversionOptions, OcrPageData
from pdf2ppt.pipeline import analyze_page, prepare_batched_ocr_page_data
from pdf2ppt.timeouts import (
    _TimeoutLane,
    call_with_timeout,
    page_timeout_seconds_from_environ,
    reset_timeout_lane_for_tests,
)


class CallWithTimeoutTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_timeout_lane_for_tests()

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

    def test_persistent_lane_reuses_same_thread(self) -> None:
        """Multi-batch OCR requires the same worker thread across calls (T1)."""

        lane = _TimeoutLane(name="affinity-test")
        thread_ids: list[int] = []

        def record_thread() -> int:
            thread_ids.append(threading.get_ident())
            return threading.get_ident()

        first = call_with_timeout(record_thread, 2.0, label="a", lane=lane)
        second = call_with_timeout(record_thread, 2.0, label="b", lane=lane)
        third = call_with_timeout(record_thread, 2.0, label="c", lane=lane)
        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(len(set(thread_ids)), 1)

    def test_lane_replaced_after_timeout_still_daemon(self) -> None:
        lane = _TimeoutLane(name="replace-test")
        hang = threading.Event()

        def blocked() -> None:
            hang.wait(timeout=30.0)

        with self.assertRaises(TimeoutError):
            call_with_timeout(blocked, 0.1, label="hang", lane=lane)

        # Replacement lane must still accept work.
        self.assertEqual(call_with_timeout(lambda: 7, 1.0, label="after", lane=lane), 7)
        hang.set()

    def test_page_timeout_env_honoured_when_options_none(self) -> None:
        with patch.dict("os.environ", {"PDF2PPT_PAGE_TIMEOUT_S": "3.5"}, clear=False):
            self.assertAlmostEqual(page_timeout_seconds_from_environ(), 3.5, places=5)
        with patch.dict("os.environ", {"PDF2PPT_PAGE_TIMEOUT_S": "0"}, clear=False):
            self.assertEqual(page_timeout_seconds_from_environ(), 0.0)


class PageOcrTimeoutFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_timeout_lane_for_tests()

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

    def test_batched_ocr_timeout_retries_per_page(self) -> None:
        import fitz

        document = fitz.open()
        document.new_page(width=100, height=100)
        document.new_page(width=100, height=100)
        options = ConversionOptions(
            input_path=MagicMock(),
            output_path=MagicMock(),
            report_path=MagicMock(),
            page_timeout_s=0.25,
            ocr_batch_size=2,
            vector_mode="off",
        )
        ocr_engine = MagicMock()
        per_page_calls = {"n": 0}

        def slow_batch(images: list, numbers: list) -> list[OcrPageData]:
            time.sleep(5.0)
            return [OcrPageData(blocks=[], image=img) for img in images]

        def fast_page(image: Image.Image, page_number: int) -> OcrPageData:
            per_page_calls["n"] += 1
            return OcrPageData(blocks=[], image=image)

        ocr_engine.extract_text_blocks_batch.side_effect = slow_batch
        ocr_engine.extract_text_blocks.side_effect = fast_page

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

        self.assertEqual(len(results), 2)
        self.assertEqual(per_page_calls["n"], 2)
        document.close()


class PredictLockOverlapTests(unittest.TestCase):
    def test_predict_lock_serializes_overlapping_calls(self) -> None:
        """Abandoned + next predict must not overlap (T3)."""

        from pdf2ppt import ocr as ocr_mod

        holding = threading.Event()
        release = threading.Event()
        overlap = {"seen": False}
        active = {"n": 0}
        lock = threading.Lock()

        def fake_predict(_arrays: object) -> list[object]:
            with lock:
                active["n"] += 1
                if active["n"] > 1:
                    overlap["seen"] = True
            holding.set()
            release.wait(timeout=2.0)
            with lock:
                active["n"] -= 1
            return []

        engine = MagicMock()
        engine.predict.side_effect = fake_predict
        ocr = ocr_mod.OcrEngine(
            lang="ch",
            model_root=None,
            use_doc_orientation=False,
            use_textline_orientation=False,
            use_doc_unwarping=False,
            det_thresh=None,
            det_box_thresh=None,
            drop_score=None,
        )
        ocr._engine = engine

        def first() -> None:
            ocr.extract_text_blocks(Image.new("RGB", (4, 4)), 1)

        def second() -> None:
            holding.wait(timeout=2.0)
            ocr.extract_text_blocks(Image.new("RGB", (4, 4)), 2)

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        self.assertTrue(holding.wait(timeout=2.0))
        t2.start()
        time.sleep(0.2)
        # Second must be blocked on the predict lock — no overlap yet.
        self.assertFalse(overlap["seen"])
        release.set()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
        self.assertFalse(overlap["seen"])


class ProcessExitsAfterTimeoutTests(unittest.TestCase):
    def test_subprocess_exits_after_simulated_hang(self) -> None:
        """Daemon lane must not keep the interpreter alive after a timeout (T2)."""

        script = """
import sys, time
from pdf2ppt.timeouts import call_with_timeout, reset_timeout_lane_for_tests

reset_timeout_lane_for_tests()

def hang():
    while True:
        pass  # busy-loop: would pin CPU forever if non-daemon

try:
    call_with_timeout(hang, 0.2, label="busy")
except TimeoutError:
    pass
# If workers were non-daemon, process would hang here at interpreter shutdown.
sys.exit(0)
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hang_driver.py"
            path.write_text(script, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
        )


if __name__ == "__main__":
    unittest.main()
