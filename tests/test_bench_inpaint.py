from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bench_inpaint  # noqa: E402

from pdf2ppt.core import OcrInitializationError, OcrPageData  # noqa: E402
from pdf2ppt.inpainting_engines import OpenCvFastInpaintingEngine, WhiteBoxInpaintingEngine  # noqa: E402
from pdf2ppt.models import TextBlock  # noqa: E402


def _solid_image(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color=color)


class OutsideMaskPixelDiffTests(unittest.TestCase):
    def test_identical_images_report_zero_diff(self) -> None:
        original = _solid_image((40, 30), (10, 20, 30))
        inpainted = original.copy()
        mask_array = np.zeros((30, 40), dtype=np.uint8)
        mask_array[5:15, 5:15] = 255

        stats = bench_inpaint.outside_mask_pixel_diff(original, inpainted, mask_array)

        self.assertEqual(stats["max_abs_diff"], 0)
        self.assertEqual(stats["changed_pixel_count"], 0)
        self.assertEqual(stats["changed_pixel_ratio"], 0.0)

    def test_change_outside_mask_is_detected(self) -> None:
        original = _solid_image((40, 30), (10, 20, 30))
        mutated = np.array(original, dtype=np.uint8)
        mutated[0, 0] = (200, 200, 200)  # a single pixel far from the mask
        inpainted = Image.fromarray(mutated, mode="RGB")
        mask_array = np.zeros((30, 40), dtype=np.uint8)
        mask_array[5:15, 5:15] = 255

        stats = bench_inpaint.outside_mask_pixel_diff(original, inpainted, mask_array)

        self.assertGreater(stats["max_abs_diff"], 0)
        self.assertEqual(stats["changed_pixel_count"], 1)

    def test_change_inside_mask_is_ignored(self) -> None:
        original = _solid_image((40, 30), (10, 20, 30))
        mutated = np.array(original, dtype=np.uint8)
        mutated[6, 6] = (255, 255, 255)  # inside the mask region below
        inpainted = Image.fromarray(mutated, mode="RGB")
        mask_array = np.zeros((30, 40), dtype=np.uint8)
        mask_array[5:15, 5:15] = 255

        stats = bench_inpaint.outside_mask_pixel_diff(original, inpainted, mask_array)

        self.assertEqual(stats["max_abs_diff"], 0)
        self.assertEqual(stats["changed_pixel_count"], 0)

    def test_empty_mask_has_no_outside_pixels_excluded(self) -> None:
        original = _solid_image((10, 10), (0, 0, 0))
        mutated = np.full((10, 10, 3), 5, dtype=np.uint8)
        inpainted = Image.fromarray(mutated, mode="RGB")
        mask_array = np.zeros((10, 10), dtype=np.uint8)  # nothing masked -> everything is "outside"

        stats = bench_inpaint.outside_mask_pixel_diff(original, inpainted, mask_array)

        self.assertEqual(stats["outside_pixel_count"], 100)
        self.assertEqual(stats["changed_pixel_count"], 100)


class AssertOutsideMaskUnchangedTests(unittest.TestCase):
    def test_passes_when_nothing_outside_mask_changed(self) -> None:
        original = _solid_image((20, 20), (50, 60, 70))
        inpainted = original.copy()
        mask_array = np.zeros((20, 20), dtype=np.uint8)
        mask_array[2:8, 2:8] = 255

        stats = bench_inpaint.assert_outside_mask_unchanged(original, inpainted, mask_array)
        self.assertEqual(stats["max_abs_diff"], 0)

    def test_raises_when_a_pixel_outside_mask_is_modified(self) -> None:
        original = _solid_image((20, 20), (50, 60, 70))
        mutated = np.array(original, dtype=np.uint8)
        mutated[15, 15] = (0, 0, 0)  # well outside the mask region below
        inpainted = Image.fromarray(mutated, mode="RGB")
        mask_array = np.zeros((20, 20), dtype=np.uint8)
        mask_array[2:8, 2:8] = 255

        with self.assertRaises(AssertionError):
            bench_inpaint.assert_outside_mask_unchanged(original, inpainted, mask_array)

    def test_tolerance_permits_small_deviations(self) -> None:
        original = _solid_image((20, 20), (50, 60, 70))
        mutated = np.array(original, dtype=np.uint8)
        mutated[15, 15] = (52, 60, 70)  # +2 in one channel, outside the mask
        inpainted = Image.fromarray(mutated, mode="RGB")
        mask_array = np.zeros((20, 20), dtype=np.uint8)
        mask_array[2:8, 2:8] = 255

        with self.assertRaises(AssertionError):
            bench_inpaint.assert_outside_mask_unchanged(original, inpainted, mask_array, tolerance=0)
        # But it is accepted once tolerance covers the deviation.
        bench_inpaint.assert_outside_mask_unchanged(original, inpainted, mask_array, tolerance=2)

    def test_real_engines_leave_correctly_scoped_pixels_untouched(self) -> None:
        # White-box has a byte-exact "everything outside the mask is untouched"
        # contract; this is the actual regression guard the benchmark relies on.
        rng = np.random.default_rng(0)
        page_array = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
        page_image = Image.fromarray(page_array, mode="RGB")
        mask_array = np.zeros((60, 80), dtype=np.uint8)
        mask_array[10:30, 15:50] = 255
        mask_image = Image.fromarray(mask_array, mode="L")

        inpainted = WhiteBoxInpaintingEngine().inpaint(page_image, mask_image)
        bench_inpaint.assert_outside_mask_unchanged(page_image, inpainted, mask_array)


class MaskBoundaryDiscontinuityTests(unittest.TestCase):
    def test_empty_mask_returns_zero(self) -> None:
        image = _solid_image((30, 30), (10, 10, 10))
        mask_array = np.zeros((30, 30), dtype=np.uint8)
        self.assertEqual(bench_inpaint.mask_boundary_discontinuity(image, mask_array), 0.0)

    def test_flat_image_has_low_discontinuity(self) -> None:
        image = _solid_image((60, 60), (128, 128, 128))
        mask_array = np.zeros((60, 60), dtype=np.uint8)
        mask_array[20:40, 20:40] = 255

        score = bench_inpaint.mask_boundary_discontinuity(image, mask_array)
        self.assertLess(score, 1.0)

    def test_harsh_seam_scores_higher_than_smooth_context(self) -> None:
        # A textured context with a flat-white patch dropped in the middle is
        # exactly the "white-box on a busy background" case: a visible seam.
        rng = np.random.default_rng(1)
        noisy = rng.integers(80, 176, size=(80, 80), dtype=np.uint8)
        textured = np.stack([noisy] * 3, axis=-1)
        harsh = textured.copy()
        harsh[25:55, 25:55] = 255
        harsh_image = Image.fromarray(harsh, mode="RGB")

        mask_array = np.zeros((80, 80), dtype=np.uint8)
        mask_array[25:55, 25:55] = 255

        seam_score = bench_inpaint.mask_boundary_discontinuity(harsh_image, mask_array)
        blended_score = bench_inpaint.mask_boundary_discontinuity(Image.fromarray(textured, mode="RGB"), mask_array)
        self.assertGreater(seam_score, blended_score)
        self.assertGreater(seam_score, 1.0)


class ResidualTextDetectionRateTests(unittest.TestCase):
    class _StubOcrEngine:
        def __init__(self, blocks: list[TextBlock], image: Image.Image) -> None:
            self._blocks = blocks
            self._image = image

        def extract_text_blocks(self, image: Image.Image, page_number: int) -> OcrPageData:
            return OcrPageData(blocks=self._blocks, image=self._image)

    def test_full_overlap_yields_rate_one(self) -> None:
        import fitz

        image = _solid_image((100, 100), (255, 255, 255))
        page_rect = fitz.Rect(0, 0, 100, 100)
        mask_array = np.zeros((100, 100), dtype=np.uint8)
        mask_array[10:30, 10:60] = 255

        detected_block = TextBlock(
            id="det_1", source="ocr", bbox=(10, 10, 60, 30), text="leftover", confidence=0.9
        )
        stub = self._StubOcrEngine([detected_block], image)

        rate = bench_inpaint.residual_text_detection_rate(stub, image, page_rect, mask_array)
        self.assertEqual(rate, 1.0)

    def test_no_detections_yields_rate_zero(self) -> None:
        import fitz

        image = _solid_image((100, 100), (255, 255, 255))
        page_rect = fitz.Rect(0, 0, 100, 100)
        mask_array = np.zeros((100, 100), dtype=np.uint8)
        mask_array[10:30, 10:60] = 255

        stub = self._StubOcrEngine([], image)

        rate = bench_inpaint.residual_text_detection_rate(stub, image, page_rect, mask_array)
        self.assertEqual(rate, 0.0)

    def test_partial_overlap_is_proportional(self) -> None:
        import fitz

        image = _solid_image((100, 100), (255, 255, 255))
        page_rect = fitz.Rect(0, 0, 100, 100)
        mask_array = np.zeros((100, 100), dtype=np.uint8)
        mask_array[0:20, 0:20] = 255  # 400 masked pixels

        # Detection covers roughly the left half of the masked region. The
        # exact expected ratio (0.55, not 0.5) follows from build_text_mask_image
        # drawing rectangles with PIL's inclusive-on-both-ends semantics, which
        # widens a 0..10 x 0..20 bbox to an 11x21 px rectangle before it is
        # intersected with the 20x20 mask.
        detected_block = TextBlock(id="det_1", source="ocr", bbox=(0, 0, 10, 20), text="x", confidence=0.9)
        stub = self._StubOcrEngine([detected_block], image)

        rate = bench_inpaint.residual_text_detection_rate(stub, image, page_rect, mask_array)
        self.assertAlmostEqual(rate, 0.55, places=2)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)


class ProbeOcrEngineTests(unittest.TestCase):
    def test_initialization_failure_degrades_gracefully(self) -> None:
        from unittest.mock import patch

        def _raise_init_error(self, image: Image.Image, page_number: int) -> OcrPageData:
            raise OcrInitializationError("no network access in sandbox")

        with patch("pdf2ppt.ocr.OcrEngine.extract_text_blocks", _raise_init_error):
            availability = bench_inpaint.probe_ocr_engine(timeout_seconds=5)

        self.assertIsNone(availability.engine)
        self.assertIn("no network access", availability.reason or "")

    def test_successful_probe_returns_engine(self) -> None:
        from unittest.mock import patch

        def _succeed(self, image: Image.Image, page_number: int) -> OcrPageData:
            return OcrPageData(blocks=[], image=image)

        with patch("pdf2ppt.ocr.OcrEngine.extract_text_blocks", _succeed):
            availability = bench_inpaint.probe_ocr_engine(timeout_seconds=5)

        self.assertIsNotNone(availability.engine)
        self.assertIsNone(availability.reason)


class BuildEngineTests(unittest.TestCase):
    def test_resolves_white_box(self) -> None:
        engine = bench_inpaint.build_engine("white-box")
        self.assertIsInstance(engine, WhiteBoxInpaintingEngine)

    def test_resolves_opencv_fast(self) -> None:
        engine = bench_inpaint.build_engine("opencv-fast")
        self.assertIsInstance(engine, OpenCvFastInpaintingEngine)


class LoadFixtureCaseTests(unittest.TestCase):
    def test_loads_solid_background_fixture(self) -> None:
        pdf_path = REPO_ROOT / "tests" / "fixtures" / "solid_background.pdf"
        case = bench_inpaint.load_fixture_case(pdf_path, dpi=96, padding_px=6)

        self.assertGreater(case.text_block_count, 0)
        self.assertGreater(case.mask_area_ratio, 0.0)
        self.assertLess(case.mask_area_ratio, 1.0)
        self.assertEqual(case.mask_array.shape[::-1], case.page_image.size)

    def test_large_mask_fixture_exceeds_auto_fallback_threshold(self) -> None:
        # This fixture exists specifically to exercise the > 0.12 auto
        # white-box fallback threshold in inpainting_overlay.py.
        pdf_path = REPO_ROOT / "tests" / "fixtures" / "large_mask.pdf"
        case = bench_inpaint.load_fixture_case(pdf_path, dpi=96, padding_px=6)
        self.assertGreater(case.mask_area_ratio, 0.12)


class RunBenchmarkSmokeTests(unittest.TestCase):
    def test_runs_end_to_end_without_ocr(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = bench_inpaint.BenchmarkConfig(
                fixtures_dir=REPO_ROOT / "tests" / "fixtures",
                engines=("white-box", "opencv-fast"),
                dpi=72,
                padding_px=6,
                ocr_mode="off",
                ocr_lang="en",
                ocr_model_root=None,
                strict_engines=("white-box",),
                json_out=tmp_path / "bench.json",
                grid_dir=tmp_path / "grids",
            )
            report = bench_inpaint.run_benchmark(config)

        self.assertFalse(report["ocr"]["available"])
        self.assertEqual(len(report["cases"]), 8)
        for case in report["cases"]:
            for engine_name in config.engines:
                metrics = case["engines"][engine_name]
                self.assertIn("seconds_per_page", metrics)
                self.assertIn("boundary_discontinuity", metrics)
                self.assertIsNone(metrics["residual_text_detection_rate"])


if __name__ == "__main__":
    unittest.main()
