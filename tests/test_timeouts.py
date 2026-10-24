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

from pdf2ppt.core import ConversionOptions, OcrPageData, OcrProcessingError
from pdf2ppt.pipeline import analyze_page, prepare_batched_ocr_page_data
from pdf2ppt.timeouts import (
    _TimeoutLane,
    abandoned_calls_running,
    call_with_timeout,
    ocr_init_timeout_seconds,
    page_timeout_seconds_from_environ,
    reset_timeout_lane_for_tests,
    timeout_lane_generation,
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

    def test_clock_starts_when_worker_begins_not_at_enqueue(self) -> None:
        """A prior slow call must not consume the next job's timeout budget."""

        lane = _TimeoutLane(name="clock-test")
        release_first = threading.Event()
        second_started = threading.Event()
        second_elapsed = {"s": 0.0}

        def first() -> str:
            release_first.wait(timeout=5.0)
            return "a"

        def second() -> str:
            t0 = time.perf_counter()
            second_started.set()
            time.sleep(0.35)
            second_elapsed["s"] = time.perf_counter() - t0
            return "b"

        results: dict[str, object] = {}
        errors: dict[str, BaseException] = {}

        def run_first() -> None:
            try:
                results["a"] = call_with_timeout(first, 2.0, label="first", lane=lane)
            except BaseException as error:  # noqa: BLE001
                errors["a"] = error

        def run_second() -> None:
            try:
                # 0.8s budget is enough for the 0.35s body *if* queue wait is excluded.
                results["b"] = call_with_timeout(second, 0.8, label="second", lane=lane)
            except BaseException as error:  # noqa: BLE001
                errors["b"] = error

        t1 = threading.Thread(target=run_first)
        t2 = threading.Thread(target=run_second)
        t1.start()
        time.sleep(0.05)
        t2.start()
        time.sleep(0.2)  # ensure second is queued behind first
        release_first.set()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        self.assertEqual(errors, {})
        self.assertEqual(results.get("a"), "a")
        self.assertEqual(results.get("b"), "b")
        self.assertTrue(second_started.is_set())
        self.assertLess(second_elapsed["s"], 0.75)

    def test_nested_call_runs_inline(self) -> None:
        lane = _TimeoutLane(name="nested-test")

        def outer() -> int:
            return call_with_timeout(lambda: 11, 1.0, label="inner", lane=lane)

        self.assertEqual(call_with_timeout(outer, 2.0, label="outer", lane=lane), 11)

    def test_thread_count_bounded_after_many_timeouts(self) -> None:
        """Abandoned workers must retire once the hung call returns (not leak forever)."""

        lane = _TimeoutLane(name="bound-test")

        def briefly_hung() -> None:
            # Longer than the timeout so callers abandon, but finite so workers retire.
            time.sleep(0.35)

        before = threading.active_count()
        for index in range(25):
            with self.assertRaises(TimeoutError):
                call_with_timeout(briefly_hung, 0.05, label=f"hang-{index}", lane=lane)
        # Wait for abandoned bodies to finish and generation-check-exit.
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if threading.active_count() <= before + 3:
                break
            time.sleep(0.05)
        settled = threading.active_count()
        self.assertLessEqual(
            settled - before,
            3,
            msg=f"before={before} settled={settled} (workers should retire after hung calls return)",
        )
        # Lane still usable.
        self.assertEqual(call_with_timeout(lambda: 1, 1.0, label="after", lane=lane), 1)

    def test_lane_replaced_after_timeout_still_works(self) -> None:
        lane = _TimeoutLane(name="replace-test")
        hang = threading.Event()

        def blocked() -> None:
            hang.wait(timeout=30.0)

        with self.assertRaises(TimeoutError):
            call_with_timeout(blocked, 0.1, label="hang", lane=lane)

        self.assertEqual(call_with_timeout(lambda: 7, 1.0, label="after", lane=lane), 7)
        hang.set()

    def test_page_timeout_env_honoured_when_options_none(self) -> None:
        with patch.dict("os.environ", {"PDF2PPT_PAGE_TIMEOUT_S": "3.5"}, clear=False):
            self.assertAlmostEqual(page_timeout_seconds_from_environ(), 3.5, places=5)
        with patch.dict("os.environ", {"PDF2PPT_PAGE_TIMEOUT_S": "0"}, clear=False):
            self.assertEqual(page_timeout_seconds_from_environ(), 0.0)

    def test_ocr_init_timeout_is_at_least_300(self) -> None:
        self.assertAlmostEqual(ocr_init_timeout_seconds(5.0), 300.0, places=5)
        self.assertAlmostEqual(ocr_init_timeout_seconds(400.0), 400.0, places=5)
        self.assertEqual(ocr_init_timeout_seconds(0), 0)
        self.assertIsNone(ocr_init_timeout_seconds(None))


class PageOcrTimeoutFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_timeout_lane_for_tests()

    def test_analyze_page_ocr_timeout_uses_empty_ocr(self) -> None:
        import pymupdf as fitz

        document = fitz.open()
        page = document.new_page(width=200, height=200)
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
        ocr_engine.warm_up = MagicMock()

        with patch("pdf2ppt.pipeline.render_page_image", return_value=Image.new("RGB", (20, 20))):
            result = analyze_page(page, options, ocr_engine)

        self.assertEqual(result.text_blocks, [])
        self.assertEqual(result.background_mode, "full-page")
        document.close()

    def test_batched_ocr_timeout_retries_per_page(self) -> None:
        import pymupdf as fitz

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
        ocr_engine.warm_up = MagicMock()

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
                        with patch("pdf2ppt.pipeline.wait_for_predict_idle", return_value=True):
                            results = prepare_batched_ocr_page_data(document, options, ocr_engine)

        self.assertEqual(len(results), 2)
        self.assertEqual(per_page_calls["n"], 2)
        ocr_engine.warm_up.assert_called()
        document.close()

    def test_wedged_predict_skips_remaining_pages_quickly(self) -> None:
        """Abandoned timed-out predict holding the lock → skip remaining OCR after one wait."""

        import pymupdf as fitz

        document = fitz.open()
        for _ in range(4):
            document.new_page(width=80, height=80)
        options = ConversionOptions(
            input_path=MagicMock(),
            output_path=MagicMock(),
            report_path=MagicMock(),
            page_timeout_s=0.2,
            ocr_batch_size=2,
            vector_mode="off",
        )
        ocr_engine = MagicMock()
        ocr_engine.warm_up = MagicMock()
        ocr_engine.extract_text_blocks_batch.side_effect = AssertionError("must not call OCR when wedged")

        idle_calls = {"n": 0}

        def never_idle(_timeout: float) -> bool:
            idle_calls["n"] += 1
            return False

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
                        with patch("pdf2ppt.pipeline.wait_for_predict_idle", side_effect=never_idle):
                            with patch("pdf2ppt.pipeline.abandoned_calls_running", return_value=1):
                                t0 = time.perf_counter()
                                results = prepare_batched_ocr_page_data(document, options, ocr_engine)
                                elapsed = time.perf_counter() - t0

        self.assertEqual(len(results), 4)
        self.assertTrue(all(data.blocks == [] for _img, data in results.values()))
        # One wedge wait (~0.2s) then fast-skip remaining batches — not 4× timeout.
        self.assertLess(elapsed, 1.5)
        self.assertEqual(idle_calls["n"], 1)
        document.close()

    def test_busy_lock_without_abandoned_call_is_not_wedged(self) -> None:
        """N3: another job's slow-but-live predict must not blank this job's OCR."""

        import pymupdf as fitz

        document = fitz.open()
        for _ in range(2):
            document.new_page(width=80, height=80)
        options = ConversionOptions(
            input_path=MagicMock(),
            output_path=MagicMock(),
            report_path=MagicMock(),
            page_timeout_s=0.2,
            ocr_batch_size=2,
            vector_mode="off",
        )
        ocr_engine = MagicMock()
        ocr_engine.warm_up = MagicMock()
        ocr_engine.extract_text_blocks_batch.side_effect = lambda images, numbers: [
            OcrPageData(blocks=[], image=image) for image in images
        ]

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
                        # Lock busy (queue wait) but no abandoned timed-out call.
                        with patch("pdf2ppt.pipeline.wait_for_predict_idle", return_value=False):
                            with patch("pdf2ppt.pipeline.abandoned_calls_running", return_value=0):
                                results = prepare_batched_ocr_page_data(document, options, ocr_engine)

        self.assertEqual(len(results), 2)
        ocr_engine.extract_text_blocks_batch.assert_called()
        document.close()

    def test_ocr_processing_error_after_lane_swap_degrades(self) -> None:
        """N1: OcrProcessingError after a timeout lane replacement → empty OCR, not abort."""

        import pymupdf as fitz

        document = fitz.open()
        document.new_page(width=80, height=80)
        options = ConversionOptions(
            input_path=MagicMock(),
            output_path=MagicMock(),
            report_path=MagicMock(),
            page_timeout_s=5.0,
            ocr_batch_size=1,
            vector_mode="off",
        )
        ocr_engine = MagicMock()
        ocr_engine.warm_up = MagicMock()
        ocr_engine.extract_text_blocks_batch.side_effect = OcrProcessingError(
            "OCR batch prediction failed: std::exception"
        )

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
                        with patch("pdf2ppt.pipeline.wait_for_predict_idle", return_value=True):
                            # Simulate a prior timeout that replaced the worker.
                            with patch("pdf2ppt.pipeline.timeout_lane_generation", side_effect=[1, 1, 2]):
                                results = prepare_batched_ocr_page_data(document, options, ocr_engine)

        self.assertEqual(len(results), 1)
        _image, page_data = results[1]
        self.assertEqual(page_data.blocks, [])
        document.close()

    def test_ocr_processing_error_when_lane_already_swapped_degrades(self) -> None:
        """N1: A later job must degrade when generation is already > 1 from an earlier swap."""

        import pymupdf as fitz

        document = fitz.open()
        document.new_page(width=80, height=80)
        options = ConversionOptions(
            input_path=MagicMock(),
            output_path=MagicMock(),
            report_path=MagicMock(),
            page_timeout_s=5.0,
            ocr_batch_size=1,
            vector_mode="off",
        )
        ocr_engine = MagicMock()
        ocr_engine.warm_up = MagicMock()
        ocr_engine.extract_text_blocks_batch.side_effect = OcrProcessingError(
            "OCR batch prediction failed: std::exception"
        )

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
                        with patch("pdf2ppt.pipeline.wait_for_predict_idle", return_value=True):
                            # Generation already > 1 for the whole job (swap happened earlier).
                            with patch("pdf2ppt.pipeline.timeout_lane_generation", return_value=2):
                                with patch("pdf2ppt.pipeline.abandoned_calls_running", return_value=0):
                                    results = prepare_batched_ocr_page_data(
                                        document, options, ocr_engine
                                    )

        self.assertEqual(len(results), 1)
        _image, page_data = results[1]
        self.assertEqual(page_data.blocks, [])
        document.close()

    def test_ocr_processing_error_without_lane_swap_still_raises(self) -> None:
        """Genuine OCR failures (no timeout / generation change) must still abort."""

        import pymupdf as fitz

        document = fitz.open()
        document.new_page(width=80, height=80)
        options = ConversionOptions(
            input_path=MagicMock(),
            output_path=MagicMock(),
            report_path=MagicMock(),
            page_timeout_s=5.0,
            ocr_batch_size=1,
            vector_mode="off",
        )
        ocr_engine = MagicMock()
        ocr_engine.warm_up = MagicMock()
        ocr_engine.extract_text_blocks_batch.side_effect = OcrProcessingError("disk full")

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
                        with patch("pdf2ppt.pipeline.wait_for_predict_idle", return_value=True):
                            with patch("pdf2ppt.pipeline.timeout_lane_generation", return_value=1):
                                with patch("pdf2ppt.pipeline.abandoned_calls_running", return_value=0):
                                    with self.assertRaises(OcrProcessingError):
                                        prepare_batched_ocr_page_data(document, options, ocr_engine)
        document.close()


class AbandonedCallAndExitDrainTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_timeout_lane_for_tests()

    def test_abandoned_calls_running_increments_on_timeout(self) -> None:
        """N3 helper: timed-out in-flight work is counted until the worker returns."""

        hang = threading.Event()

        def blocked() -> None:
            hang.wait(timeout=30.0)

        self.assertEqual(abandoned_calls_running(), 0)
        with self.assertRaises(TimeoutError):
            call_with_timeout(blocked, 0.1, label="hang")
        self.assertGreaterEqual(abandoned_calls_running(), 1)
        hang.set()
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline and abandoned_calls_running() > 0:
            time.sleep(0.02)
        self.assertEqual(abandoned_calls_running(), 0)

    def test_reset_for_tests_does_not_drive_abandoned_count_negative(self) -> None:
        """N4: reset_for_tests() zeroing must not let abandoned workers decrement below 0."""

        hang = threading.Event()

        def blocked() -> None:
            hang.wait(timeout=30.0)

        with self.assertRaises(TimeoutError):
            call_with_timeout(blocked, 0.1, label="hang")
        self.assertGreaterEqual(abandoned_calls_running(), 1)
        reset_timeout_lane_for_tests()
        self.assertEqual(abandoned_calls_running(), 0)
        hang.set()
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            count = abandoned_calls_running()
            self.assertGreaterEqual(count, 0)
            if count == 0:
                # Give the abandoned worker a moment to finish its decrement.
                time.sleep(0.05)
                self.assertGreaterEqual(abandoned_calls_running(), 0)
                break
            time.sleep(0.02)
        self.assertGreaterEqual(abandoned_calls_running(), 0)

    def test_timeout_lane_generation_bumps_on_timeout(self) -> None:
        hang = threading.Event()
        before = timeout_lane_generation()

        def blocked() -> None:
            hang.wait(timeout=30.0)

        with self.assertRaises(TimeoutError):
            call_with_timeout(blocked, 0.1, label="hang")
        self.assertGreater(timeout_lane_generation(), before)
        hang.set()

    def test_exit_drain_waits_for_abandoned_predict(self) -> None:
        """N2: atexit drain calls wait_for_predict_idle only when abandoned work remains."""

        from pdf2ppt import ocr as ocr_mod

        with patch("pdf2ppt.ocr.abandoned_calls_running", return_value=1):
            with patch("pdf2ppt.ocr.wait_for_predict_idle", return_value=True) as wait_idle:
                ocr_mod._drain_abandoned_predict_at_exit()
                wait_idle.assert_called_once_with(ocr_mod.EXIT_DRAIN_TIMEOUT_SECONDS)

        with patch("pdf2ppt.ocr.abandoned_calls_running", return_value=0):
            with patch("pdf2ppt.ocr.wait_for_predict_idle") as wait_idle:
                ocr_mod._drain_abandoned_predict_at_exit()
                wait_idle.assert_not_called()


class PredictLockOverlapTests(unittest.TestCase):
    def test_predict_lock_serializes_overlapping_calls(self) -> None:
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
        self.assertFalse(overlap["seen"])
        release.set()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
        self.assertFalse(overlap["seen"])

    def test_wait_for_predict_idle_false_while_held(self) -> None:
        from pdf2ppt import ocr as ocr_mod

        acquired = threading.Event()
        release = threading.Event()

        def holder() -> None:
            ocr_mod._PADDLE_PREDICT_LOCK.acquire()
            acquired.set()
            release.wait(timeout=5.0)
            ocr_mod._PADDLE_PREDICT_LOCK.release()

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(acquired.wait(timeout=2.0))
        self.assertFalse(ocr_mod.wait_for_predict_idle(0.15))
        release.set()
        thread.join(timeout=3.0)
        self.assertTrue(ocr_mod.wait_for_predict_idle(1.0))


class ProcessExitsAfterTimeoutTests(unittest.TestCase):
    def test_subprocess_exits_after_simulated_hang(self) -> None:
        script = """
import sys
from pdf2ppt.timeouts import call_with_timeout, reset_timeout_lane_for_tests

reset_timeout_lane_for_tests()

def hang():
    while True:
        pass

try:
    call_with_timeout(hang, 0.2, label="busy")
except TimeoutError:
    pass
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
