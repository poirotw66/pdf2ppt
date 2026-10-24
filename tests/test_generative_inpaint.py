"""Tests for the opt-in generative inpaint backend (providers mocked; no network)."""

from __future__ import annotations

import base64
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from pdf2ppt.cli import build_parser
from pdf2ppt.core import ConversionOptions
from pdf2ppt.generative_inpaint import (
    DEFAULT_FLAT_RING_TEXTURE_STD,
    DEFAULT_TEXT_REMOVAL_PROMPT,
    DEFAULT_TEXTURED_RING_TEXTURE_STD,
    GeminiImageEditProvider,
    GenerativeBudget,
    GenerativeEditRequest,
    GenerativeEditResult,
    GenerativeInpaintingEngine,
    GenerativeProviderError,
    OpenAIImageEditProvider,
    composite_masked_edit,
    compute_generative_cache_key,
    context_ring_highpass_std,
    context_ring_luma_std,
    context_ring_texture_score,
    create_provider,
    effective_low_freq_mae_threshold,
    is_expected_safety_block_message,
    low_frequency_masked_mae,
    seamless_clone_masked_edit,
    uses_generative_inpaint_engine,
)
from pdf2ppt.generative_inpaint.aspect import (
    closest_aspect_ratio,
    plan_aspect_pad,
    unpad_image_from_plan,
)
from pdf2ppt.generative_inpaint.budget import GenerativeCallRecord
from pdf2ppt.generative_inpaint.cache import GenerativeInpaintCache
from pdf2ppt.generative_inpaint.pricing import (
    GEMINI_FALLBACK_COST_USD,
    cost_from_gemini_usage,
    cost_from_openai_usage,
)
from pdf2ppt.generative_inpaint.safety import assert_unmasked_pixels_unchanged
from pdf2ppt.inpainting_masks import dilate_px_for_line_height, dilate_text_mask_by_line_height
from pdf2ppt.inpainting_overlay import resolve_background_inpainting_engine
from pdf2ppt.models import ConversionReport, PageResult, QualityScore, TextBlock


def _solid_rgb(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def _rect_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    x0, y0, x1, y1 = box
    for y in range(y0, y1):
        for x in range(x0, x1):
            mask.putpixel((x, y), 255)
    return mask


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _textured_rgb(size: tuple[int, int]) -> Image.Image:
    """High-variance background so the flat-ring API skip does not fire."""

    width, height = size
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = ((x * 37 + y * 91) % 255, (x * 17 + y * 53) % 255, (x * 7 + y * 29) % 255)
    return Image.fromarray(arr, mode="RGB")


def _linear_gradient_rgb(size: tuple[int, int]) -> Image.Image:
    width, height = size
    ramp = np.linspace(20, 220, width, dtype=np.float32)
    gray = np.tile(ramp, (height, 1))
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1).astype(np.uint8), mode="RGB")


def _radial_gradient_rgb(size: tuple[int, int]) -> Image.Image:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    gray = (radius / max(float(radius.max()), 1e-6) * 235.0).astype(np.float32)
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1).astype(np.uint8), mode="RGB")


def _center_mask(size: tuple[int, int], frac: float = 0.5) -> Image.Image:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    y0 = int(height * (1.0 - frac) / 2.0)
    y1 = int(height * (1.0 + frac) / 2.0)
    x0 = int(width * (1.0 - frac) / 2.0)
    x1 = int(width * (1.0 + frac) / 2.0)
    mask[y0:y1, x0:x1] = 255
    return Image.fromarray(mask, mode="L")


def _engine_kwargs(**overrides: object) -> dict[str, object]:
    """Defaults that keep unit tests on the generative path (no flat-ring skip)."""

    base: dict[str, object] = {
        "cache_dir": None,
        "verify_ocr": False,
        "low_freq_mae_threshold": -1.0,
        "skip_flat_ring": False,
    }
    base.update(overrides)
    return base


class FakeProvider:
    name = "fake"
    default_model = "fake-model"

    def __init__(
        self,
        *,
        fill_color: tuple[int, int, int] = (10, 20, 30),
        fail: bool = False,
        fail_error: GenerativeProviderError | None = None,
        cost_usd: float = 0.05,
        cost_uncertain: bool = False,
        cost_source: str = "estimate",
    ) -> None:
        self.fill_color = fill_color
        self.fail = fail
        self.fail_error = fail_error
        self.cost_usd = cost_usd
        self.cost_uncertain = cost_uncertain
        self.cost_source = cost_source
        self.calls = 0
        self._configured = True

    def is_configured(self) -> bool:
        return self._configured

    def estimate_cost_usd(self, *, model: str, width: int, height: int) -> float:
        del model, width, height
        return self.cost_usd

    def edit(self, request: GenerativeEditRequest) -> GenerativeEditResult:
        self.calls += 1
        if self.fail:
            if self.fail_error is not None:
                raise self.fail_error
            raise GenerativeProviderError("simulated provider failure")
        # Paint the whole crop a solid color; compositor must keep unmasked pixels.
        edited = Image.new("RGB", request.image.size, self.fill_color)
        return GenerativeEditResult(
            image=edited,
            model=self.default_model,
            provider_name=self.name,
            estimated_cost_usd=self.cost_usd,
            cost_source=self.cost_source,
            cost_uncertain=self.cost_uncertain,
            raw_response_note="fake",
        )


class GenerativeHelpersTests(unittest.TestCase):
    def test_uses_generative_engine_names(self) -> None:
        self.assertTrue(uses_generative_inpaint_engine("generative-openai"))
        self.assertTrue(uses_generative_inpaint_engine("generative-gemini"))
        self.assertFalse(uses_generative_inpaint_engine("opencv-fast"))
        self.assertFalse(uses_generative_inpaint_engine("auto"))

    def test_default_prompt_is_neutral(self) -> None:
        lowered = DEFAULT_TEXT_REMOVAL_PROMPT.lower()
        self.assertNotIn("watermark", lowered)
        self.assertNotIn("logo", lowered)
        self.assertIn("background", lowered)

    def test_composite_preserves_unmasked_pixels(self) -> None:
        source = _solid_rgb((32, 32), (200, 100, 50))
        # Structured edit so Poisson/seamlessClone cannot reconstruct the
        # destination solid color from zero source gradients alone.
        edited_arr = np.array(source)
        for y in range(8, 24):
            edited_arr[y, 8:24] = (y * 3, 20, 255 - y * 3)
        edited = Image.fromarray(edited_arr, mode="RGB")
        mask = _rect_mask((32, 32), (8, 8, 24, 24))
        result = composite_masked_edit(source, edited, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        source_arr = np.array(source)
        result_arr = np.array(result)
        mask_arr = np.array(mask) > 0
        self.assertFalse(np.array_equal(source_arr[mask_arr], result_arr[mask_arr]))

    def test_seamless_clone_preserves_unmasked_pixels(self) -> None:
        source = np.zeros((48, 48, 3), dtype=np.uint8)
        for y in range(48):
            source[y, :, :] = (y * 4, 80, 200 - y * 2)
        edited = source.copy()
        edited[16:32, 16:32] = (40, 40, 40)
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[16:32, 16:32] = 255
        blended = seamless_clone_masked_edit(source, edited, mask)
        self.assertTrue(np.array_equal(blended[mask == 0], source[mask == 0]))

    def test_normal_clone_does_not_restore_original_text_edges(self) -> None:
        """Blocker regression: MIXED_CLONE pulled glyph edges back into clean fills."""

        source = np.full((64, 64, 3), 180, dtype=np.uint8)
        # Synthetic dark "glyphs" inside the mask (high edge energy).
        source[20:44, 22:26] = (10, 10, 10)
        source[20:44, 38:42] = (10, 10, 10)
        source[28:32, 22:42] = (10, 10, 10)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        # Clean generative fill (no glyphs).
        edited = np.full((64, 64, 3), 180, dtype=np.uint8)
        edited[16:48, 16:48] = (175, 175, 175)

        normal = seamless_clone_masked_edit(source, edited, mask, clone_mode=cv2.NORMAL_CLONE)
        mixed = seamless_clone_masked_edit(source, edited, mask, clone_mode=cv2.MIXED_CLONE)
        default = seamless_clone_masked_edit(source, edited, mask)

        def edge_energy(rgb: np.ndarray) -> float:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            return float(np.mean(edges[mask > 0]))

        # Default must match NORMAL (not MIXED): interior edges stay low.
        self.assertLess(edge_energy(default), edge_energy(mixed) * 0.5)
        self.assertAlmostEqual(edge_energy(default), edge_energy(normal), delta=1.0)
        self.assertLess(edge_energy(normal), 5.0)

    def test_low_frequency_masked_mae_detects_bright_patch(self) -> None:
        left = np.full((32, 32, 3), 100, dtype=np.uint8)
        right = np.full((32, 32, 3), 100, dtype=np.uint8)
        right[8:24, 8:24] = 220
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        mae = low_frequency_masked_mae(left, right, mask, blur_sigma=8.0)
        self.assertGreater(mae, 40.0)

    def test_highpass_ring_classifies_gradients_flat_and_noise_textured(self) -> None:
        size = (128, 128)
        mask = np.array(_center_mask(size, frac=0.55))
        linear = np.array(_linear_gradient_rgb(size))
        radial = np.array(_radial_gradient_rgb(size))
        flat = np.full((size[1], size[0], 3), 180, dtype=np.uint8)
        noisy = np.array(_textured_rgb(size))

        # Raw luma std still sees full-ramp gradients as "textured".
        self.assertGreater(context_ring_luma_std(linear, mask), 20.0)
        self.assertGreater(context_ring_luma_std(radial, mask), 10.0)

        for source in (linear, radial, flat):
            score = context_ring_texture_score(source, mask, measure="highpass")
            self.assertLessEqual(score, DEFAULT_FLAT_RING_TEXTURE_STD)
            self.assertLessEqual(context_ring_highpass_std(source, mask), DEFAULT_FLAT_RING_TEXTURE_STD)

        textured_score = context_ring_texture_score(noisy, mask, measure="highpass")
        self.assertGreater(textured_score, DEFAULT_TEXTURED_RING_TEXTURE_STD)
        self.assertIsNone(
            effective_low_freq_mae_threshold(4.0, textured_score, textured_std=DEFAULT_TEXTURED_RING_TEXTURE_STD)
        )

    def test_highpass_textured_threshold_calibrated_to_v5_bench(self) -> None:
        """v5 Gemini bench: flat ≤ 1.04, photo_texture 6.27, geometric patch 14.91."""

        # Soft photo ring must disable the low-freq gate (legacy textured=16 did not).
        self.assertIsNone(
            effective_low_freq_mae_threshold(
                4.0,
                6.27,
                textured_std=DEFAULT_TEXTURED_RING_TEXTURE_STD,
                scale_ref_std=2.0,
            )
        )
        # Mid geometric_chart patch (~14.9) is clearly textured → gate off.
        self.assertIsNone(
            effective_low_freq_mae_threshold(
                4.0,
                14.91,
                textured_std=DEFAULT_TEXTURED_RING_TEXTURE_STD,
                scale_ref_std=2.0,
            )
        )
        # Gradient rings stay below flat; mid-band (2–4) still scales the gate.
        self.assertLessEqual(1.04, DEFAULT_FLAT_RING_TEXTURE_STD)
        scaled = effective_low_freq_mae_threshold(
            4.0,
            3.0,
            textured_std=DEFAULT_TEXTURED_RING_TEXTURE_STD,
            scale_ref_std=2.0,
        )
        self.assertIsNotNone(scaled)
        assert scaled is not None
        self.assertAlmostEqual(scaled, 6.0, places=5)

    def test_cache_roundtrip(self) -> None:
        import tempfile
        import uuid

        with tempfile.TemporaryDirectory(prefix="pdf2ppt-gen-cache-") as tmp:
            cache = GenerativeInpaintCache(Path(tmp))
            # Unique pixels so the key cannot collide with leftover caches.
            image = _solid_rgb((16, 16), (1, 2, 3))
            image.putpixel((0, 0), (ord(uuid.uuid4().hex[0]) % 256, 4, 5))
            mask = _rect_mask((16, 16), (2, 2, 8, 8))
            key = compute_generative_cache_key(
                provider="fake",
                model="fake-model",
                image=image,
                mask_image=mask,
                prompt=DEFAULT_TEXT_REMOVAL_PROMPT,
            )
            self.assertIsNone(cache.get(key))
            cache.put(key, image)
            loaded = cache.get(key)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.size, image.size)


class AspectAndPricingTests(unittest.TestCase):
    def test_closest_aspect_ratio_for_wide_crop(self) -> None:
        self.assertEqual(closest_aspect_ratio(1698, 462), "4:1")
        self.assertEqual(closest_aspect_ratio(1280, 720), "16:9")
        self.assertEqual(closest_aspect_ratio(100, 100), "1:1")

    def test_unpad_does_not_stretch_content(self) -> None:
        plan = plan_aspect_pad(100, 40)  # 2.5:1 → nearest documented ratio is 21:9
        self.assertEqual(plan.aspect_ratio, "21:9")
        self.assertGreater(plan.padded_size[0] * plan.padded_size[1], 100 * 40)
        # Model returned a correct-AR canvas with a solid color; unpad must keep size.
        padded = Image.new("RGB", plan.padded_size, (10, 20, 30))
        left, top, right, bottom = plan.content_box
        for y in range(top, bottom):
            for x in range(left, right):
                padded.putpixel((x, y), (200, 100, 50))
        restored = unpad_image_from_plan(padded, plan)
        self.assertEqual(restored.size, (100, 40))
        arr = np.array(restored)
        self.assertTrue(np.all(arr == (200, 100, 50)))

    def test_gemini_usage_cost_matches_measured_ballpark(self) -> None:
        usage = {
            "total_input_tokens": 1200,
            "total_thought_tokens": 300,
            "total_output_tokens": 1120,
            "output_tokens_by_modality": [{"modality": "image", "tokens": 1120}],
        }
        cost = cost_from_gemini_usage(usage)
        assert cost is not None
        self.assertFalse(cost.uncertain)
        self.assertEqual(cost.source, "usage")
        # 1200*1.5 + 300*7.5 + 1120*30 = 1800+2250+33600 = 37650 / 1e6 = 0.03765
        self.assertAlmostEqual(cost.usd, 0.03765, places=5)

    def test_gemini_usage_prices_unlisted_output_text_tokens(self) -> None:
        usage = {
            "total_input_tokens": 1200,
            "total_thought_tokens": 300,
            "total_output_tokens": 1376,
            "output_tokens_by_modality": [{"modality": "image", "tokens": 1120}],
        }
        cost = cost_from_gemini_usage(usage)
        assert cost is not None
        # leftover 256 text tokens at $7.50/M → +0.00192 vs the 1120-only case
        self.assertAlmostEqual(cost.usd, 0.03957, places=5)

    def test_openai_usage_cost(self) -> None:
        usage = {
            "input_tokens": 1000,
            "output_tokens": 1000,
            "input_tokens_details": {"text_tokens": 200, "image_tokens": 800},
        }
        cost = cost_from_openai_usage(usage)
        assert cost is not None
        # 200*5 + 800*8 + 1000*30 = 1000+6400+30000 = 37400 / 1e6
        self.assertAlmostEqual(cost.usd, 0.0374, places=5)


class GenerativeEngineTests(unittest.TestCase):
    def test_engine_inpaints_only_mask_and_records_budget(self) -> None:
        provider = FakeProvider(fill_color=(9, 9, 9))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
        )
        source = _solid_rgb((64, 64), (180, 180, 180))
        # Draw fake "text" pixels inside the mask region so the source differs.
        source_arr = np.array(source)
        source_arr[20:40, 20:40] = (0, 0, 0)
        source = Image.fromarray(source_arr, mode="RGB")
        mask = _rect_mask((64, 64), (20, 20, 40, 40))

        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertGreaterEqual(provider.calls, 1)
        self.assertEqual(budget.calls, provider.calls)
        self.assertGreater(budget.estimated_cost_usd, 0.0)
        self.assertIn("generative", engine.name)

    def test_provider_failure_falls_back_to_local(self) -> None:
        provider = FakeProvider(fail=True)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=GenerativeBudget(max_calls=5, max_cost_usd=1.0),
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
        )
        source = _solid_rgb((48, 48), (120, 130, 140))
        mask = _rect_mask((48, 48), (10, 10, 30, 30))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 1)
        self.assertGreaterEqual(engine.budget.fallbacks, 1)

    def test_local_fallback_preserves_unmasked_even_if_opencv_touches_outside(self) -> None:
        """Regression: fallback must re-composite through the mask."""

        class TouchyFallback(FakeProvider):
            def edit(self, request: GenerativeEditRequest) -> GenerativeEditResult:
                raise GenerativeProviderError("force fallback")

        engine = GenerativeInpaintingEngine(
            provider=TouchyFallback(fail=True),
            budget=GenerativeBudget(max_calls=5, max_cost_usd=1.0),
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
        )
        # Gradient-like source so opencv-fast surface/Telea paths have work to do.
        source_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        for y in range(64):
            source_arr[y, :, :] = (y * 3, 40, 200 - y)
        source_arr[20:40, 20:40] = (255, 0, 0)
        source = Image.fromarray(source_arr, mode="RGB")
        mask = _rect_mask((64, 64), (20, 20, 40, 40))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)

    def test_safety_block_classified_separately_and_zero_cost(self) -> None:
        error = GenerativeProviderError(
            "Gemini interactions HTTP 400: Image generation blocked due to copyright/recitation. "
            "You will not be charged for this request.",
            is_expected_safety_block=True,
            may_have_been_billed=False,
            estimated_cost_usd=0.0,
        )
        provider = FakeProvider(fail=True, fail_error=error)
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
            safety_block_retries=1,
        )
        source = _solid_rgb((48, 48), (90, 90, 90))
        mask = _rect_mask((48, 48), (12, 12, 36, 36))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        # One free retry then fallback; both blocked attempts cost 0 and are not billed.
        self.assertEqual(provider.calls, 2)
        self.assertEqual(budget.attempts, 2)
        self.assertEqual(budget.billed_calls, 0)
        self.assertEqual(budget.calls, 0)
        self.assertEqual(budget.safety_block_retries, 1)
        self.assertEqual(budget.safety_block_fallbacks, 1)
        self.assertEqual(budget.api_error_fallbacks, 0)
        self.assertEqual(budget.estimated_cost_usd, 0.0)
        payload = budget.to_dict()
        self.assertEqual(payload["safety_block_fallbacks"], 1)
        self.assertEqual(payload["safety_block_retries"], 1)
        self.assertEqual(payload["attempts"], 2)
        self.assertEqual(payload["billed_calls"], 0)
        self.assertTrue(any(r.fallback_kind == "safety_block" for r in budget.records))
        self.assertIn("local-safety-block", engine.last_debug_note or "")

    def test_parse_failure_records_uncertain_cost(self) -> None:
        error = GenerativeProviderError(
            "Gemini response did not include an output image.",
            may_have_been_billed=True,
            estimated_cost_usd=GEMINI_FALLBACK_COST_USD,
            cost_uncertain=True,
        )
        provider = FakeProvider(fail=True, fail_error=error)
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
        )
        source = _solid_rgb((32, 32), (10, 20, 30))
        mask = _rect_mask((32, 32), (8, 8, 24, 24))
        engine.inpaint(source, mask)
        self.assertGreater(budget.estimated_cost_usd, 0.0)
        self.assertTrue(budget.cost_uncertain)
        self.assertEqual(budget.api_error_fallbacks, 1)

    def test_ocr_safety_rejection_falls_back(self) -> None:
        provider = FakeProvider(fill_color=(5, 5, 5))

        def reject(_image: Image.Image, _mask: Image.Image) -> bool:
            return False

        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=GenerativeBudget(max_calls=5, max_cost_usd=1.0),
            cache_dir=None,
            verify_ocr=True,
            residual_checker=reject,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
        )
        source = _solid_rgb((48, 48), (90, 90, 90))
        mask = _rect_mask((48, 48), (12, 12, 36, 36))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertGreaterEqual(engine.budget.fallbacks, 1)

    def test_low_freq_gate_falls_back_to_opencv(self) -> None:
        # Semi-flat ring keeps the gate active; bright generative fill trips it.
        provider = FakeProvider(fill_color=(240, 240, 240))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=4.0,
            skip_flat_ring=False,
            textured_ring_luma_std=64.0,  # keep gate on for this mild ring
        )
        source = _solid_rgb((48, 48), (40, 40, 40))
        source_arr = np.array(source)
        source_arr[12:36, 12:36] = (255, 0, 0)
        source = Image.fromarray(source_arr, mode="RGB")
        mask = _rect_mask((48, 48), (12, 12, 36, 36))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(budget.low_freq_fallbacks, 1)
        self.assertEqual(budget.to_dict()["low_freq_fallbacks"], 1)
        self.assertTrue(any(r.fallback_kind == "low_freq" for r in budget.records))

    def test_clean_fill_with_text_source_passes_residual_checker(self) -> None:
        """Original has glyphs; clean Telea fill + NORMAL_CLONE must not reintroduce them."""

        class CleanFillProvider(FakeProvider):
            def edit(self, request: GenerativeEditRequest) -> GenerativeEditResult:
                self.calls += 1
                img = np.array(request.image.convert("RGB"))
                mask = np.array(request.mask_image.convert("L"))
                dilated = cv2.dilate((mask > 0).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
                out = cv2.inpaint(img, dilated, 5, cv2.INPAINT_TELEA)
                return GenerativeEditResult(
                    image=Image.fromarray(out, mode="RGB"),
                    model=self.default_model,
                    provider_name=self.name,
                    estimated_cost_usd=self.cost_usd,
                    cost_source=self.cost_source,
                    cost_uncertain=self.cost_uncertain,
                    raw_response_note="telea-clean",
                )

        source = np.full((64, 64, 3), 200, dtype=np.uint8)
        source[22:42, 24:28] = (15, 15, 15)
        source[22:42, 36:40] = (15, 15, 15)
        mask = _rect_mask((64, 64), (16, 16, 48, 48))

        def residual_ok(image: Image.Image, mask_image: Image.Image) -> bool:
            arr = np.array(image.convert("RGB"))
            m = np.array(mask_image.convert("L")) > 0
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            # Original glyphs produce high edge density; a clean fill must not.
            return float(np.mean(edges[m])) < 8.0

        provider = CleanFillProvider()
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=True,
            residual_checker=residual_ok,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
        )
        result = engine.inpaint(Image.fromarray(source, mode="RGB"), mask)
        assert_unmasked_pixels_unchanged(Image.fromarray(source, mode="RGB"), result, mask)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(budget.fallbacks, 0)
        self.assertIn("generative", engine.last_debug_note or "")

    def test_flat_ring_skips_api_without_spend(self) -> None:
        provider = FakeProvider(fill_color=(9, 9, 9))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=True,
            flat_ring_luma_std=DEFAULT_FLAT_RING_TEXTURE_STD,
            ring_texture_measure="highpass",
        )
        source = _solid_rgb((48, 48), (200, 200, 200))
        source_arr = np.array(source)
        source_arr[12:36, 12:36] = (0, 0, 0)  # text inside mask only
        source = Image.fromarray(source_arr, mode="RGB")
        mask = _rect_mask((48, 48), (12, 12, 36, 36))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(budget.billed_calls, 0)
        self.assertEqual(budget.flat_ring_skips, 1)
        self.assertEqual(budget.ring_texture_measure, "highpass")
        self.assertEqual(budget.to_dict()["ring_texture_measure"], "highpass")
        self.assertIn("local-flat-ring", engine.last_debug_note or "")
        self.assertTrue(any("highpass" in (r.note or "") for r in budget.records))

    def test_gradient_rings_skip_api_with_fake_provider(self) -> None:
        for label, source in (
            ("linear", _linear_gradient_rgb((96, 96))),
            ("radial", _radial_gradient_rgb((96, 96))),
        ):
            with self.subTest(label=label):
                provider = FakeProvider(fill_color=(9, 9, 9))
                budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
                engine = GenerativeInpaintingEngine(
                    provider=provider,
                    budget=budget,
                    cache_dir=None,
                    verify_ocr=False,
                    low_freq_mae_threshold=-1.0,
                    skip_flat_ring=True,
                    ring_texture_measure="highpass",
                    flat_ring_luma_std=DEFAULT_FLAT_RING_TEXTURE_STD,
                )
                mask = _center_mask((96, 96), frac=0.5)
                result = engine.inpaint(source, mask)
                assert_unmasked_pixels_unchanged(source, result, mask)
                self.assertEqual(provider.calls, 0, msg=label)
                self.assertEqual(budget.billed_calls, 0, msg=label)
                self.assertEqual(budget.flat_ring_skips, 1, msg=label)
                self.assertIn("local-flat-ring", engine.last_debug_note or "")

    def test_photo_like_ring_keeps_generative_path(self) -> None:
        """Noisy photo-like rings must not flat-skip; fake provider is still called."""

        provider = FakeProvider(fill_color=(40, 50, 60))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=True,
            ring_texture_measure="highpass",
            flat_ring_luma_std=DEFAULT_FLAT_RING_TEXTURE_STD,
            textured_ring_luma_std=DEFAULT_TEXTURED_RING_TEXTURE_STD,
        )
        source = _textured_rgb((96, 96))
        mask = _center_mask((96, 96), frac=0.45)
        score = context_ring_texture_score(np.array(source), np.array(mask), measure="highpass")
        self.assertGreater(score, DEFAULT_FLAT_RING_TEXTURE_STD)
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(budget.flat_ring_skips, 0)
        self.assertEqual(budget.billed_calls, 1)
        self.assertIn("generative", engine.last_debug_note or "")

    def test_textured_ring_skips_low_freq_gate(self) -> None:
        # Photo-like surroundings: low-freq MAE would be high, but gate is skipped.
        provider = FakeProvider(fill_color=(10, 20, 30))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=4.0,
            skip_flat_ring=True,
            ring_texture_measure="highpass",
            textured_ring_luma_std=DEFAULT_TEXTURED_RING_TEXTURE_STD,
        )
        source = _textured_rgb((64, 64))
        source_arr = np.array(source)
        source_arr[20:44, 20:44] = (255, 0, 0)
        source = Image.fromarray(source_arr, mode="RGB")
        mask = _rect_mask((64, 64), (20, 20, 44, 44))
        ring_score = context_ring_texture_score(
            np.array(source), np.array(mask), measure="highpass"
        )
        self.assertGreaterEqual(ring_score, DEFAULT_TEXTURED_RING_TEXTURE_STD)
        self.assertIsNone(
            effective_low_freq_mae_threshold(
                4.0, ring_score, textured_std=DEFAULT_TEXTURED_RING_TEXTURE_STD
            )
        )
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(budget.low_freq_fallbacks, 0)
        self.assertEqual(budget.billed_calls, 1)
        self.assertEqual(budget.flat_ring_skips, 0)

    def test_soft_photo_highpass_keeps_generative_despite_low_freq_mae(self) -> None:
        """v5 photo_texture regression: highpass ≈ 6.27, low-freq MAE ≈ 14.33.

        With the legacy textured threshold (16), the gate stayed on at 12.54 and
        rejected Gemini after a billed call. Recalibrated textured=4 skips the gate.
        """

        provider = FakeProvider(fill_color=(90, 95, 100))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=4.0,
            skip_flat_ring=True,
            ring_texture_measure="highpass",
            flat_ring_luma_std=DEFAULT_FLAT_RING_TEXTURE_STD,
            textured_ring_luma_std=DEFAULT_TEXTURED_RING_TEXTURE_STD,
            low_freq_std_scale_ref=2.0,
        )
        source = _solid_rgb((64, 64), (120, 110, 100))
        mask = _rect_mask((64, 64), (16, 16, 48, 48))
        with (
            patch(
                "pdf2ppt.generative_inpaint.engine.context_ring_texture_score",
                return_value=6.27,
            ),
            patch(
                "pdf2ppt.generative_inpaint.engine.low_frequency_masked_mae",
                return_value=14.33,
            ),
        ):
            # Legacy textured=16 would keep a loosened gate at 12.54 and reject.
            legacy = effective_low_freq_mae_threshold(
                4.0, 6.27, textured_std=16.0, scale_ref_std=2.0
            )
            self.assertIsNotNone(legacy)
            assert legacy is not None
            self.assertAlmostEqual(legacy, 12.54, places=2)
            self.assertGreater(14.33, legacy)

            result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(budget.billed_calls, 1)
        self.assertEqual(budget.low_freq_fallbacks, 0)
        self.assertEqual(budget.flat_ring_skips, 0)
        self.assertIn("generative", engine.last_debug_note or "")

    def test_geometric_mid_highpass_patch_skips_low_freq_gate(self) -> None:
        """v5 geometric_chart patch at highpass ≈ 14.91 stays on the generative path."""

        provider = FakeProvider(fill_color=(200, 200, 200))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=4.0,
            skip_flat_ring=True,
            ring_texture_measure="highpass",
            textured_ring_luma_std=DEFAULT_TEXTURED_RING_TEXTURE_STD,
        )
        source = _solid_rgb((112, 64), (240, 240, 240))
        mask = _rect_mask((112, 64), (20, 16, 92, 48))
        with (
            patch(
                "pdf2ppt.generative_inpaint.engine.context_ring_texture_score",
                return_value=14.91,
            ),
            patch(
                "pdf2ppt.generative_inpaint.engine.low_frequency_masked_mae",
                return_value=20.0,
            ),
        ):
            result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(budget.low_freq_fallbacks, 0)
        self.assertEqual(budget.flat_ring_skips, 0)
        self.assertIn("generative", engine.last_debug_note or "")

    def test_safety_blocks_do_not_consume_billed_budget(self) -> None:
        error = GenerativeProviderError(
            "Image generation blocked due to copyright/recitation. "
            "You will not be charged for this request.",
            is_expected_safety_block=True,
            may_have_been_billed=False,
            estimated_cost_usd=0.0,
        )
        provider = FakeProvider(fail=True, fail_error=error)
        budget = GenerativeBudget(max_calls=1, max_cost_usd=1.0, max_attempts=6)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
            safety_block_retries=1,
        )
        source = _solid_rgb((32, 32), (90, 90, 90))
        mask = _rect_mask((32, 32), (8, 8, 24, 24))
        engine.inpaint(source, mask)
        # Two free attempts; billed quota (max_calls=1) remains unused.
        self.assertEqual(provider.calls, 2)
        self.assertEqual(budget.attempts, 2)
        self.assertEqual(budget.billed_calls, 0)
        self.assertEqual(budget.calls, 0)
        self.assertEqual(budget.estimated_cost_usd, 0.0)
        # A subsequent crop could still afford one billed call.
        self.assertTrue(budget.can_afford(0.05))

    def test_safety_block_retry_recovers_on_second_attempt(self) -> None:
        class FlakySafetyProvider(FakeProvider):
            def edit(self, request: GenerativeEditRequest) -> GenerativeEditResult:
                self.calls += 1
                if self.calls == 1:
                    raise GenerativeProviderError(
                        "Image generation blocked due to copyright/recitation. "
                        "You will not be charged for this request.",
                        is_expected_safety_block=True,
                        may_have_been_billed=False,
                        estimated_cost_usd=0.0,
                    )
                edited = Image.new("RGB", request.image.size, self.fill_color)
                return GenerativeEditResult(
                    image=edited,
                    model=self.default_model,
                    provider_name=self.name,
                    estimated_cost_usd=self.cost_usd,
                    cost_source=self.cost_source,
                    cost_uncertain=self.cost_uncertain,
                    raw_response_note="recovered",
                )

        provider = FlakySafetyProvider(fill_color=(12, 12, 12))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
            safety_block_retries=1,
        )
        source = _solid_rgb((40, 40), (80, 80, 80))
        mask = _rect_mask((40, 40), (8, 8, 28, 28))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(budget.attempts, 2)
        self.assertEqual(budget.billed_calls, 1)
        self.assertEqual(budget.calls, 1)
        self.assertEqual(budget.safety_block_retries, 1)
        self.assertEqual(budget.safety_block_fallbacks, 0)
        self.assertGreater(budget.estimated_cost_usd, 0.0)

    def test_budget_exhaustion_skips_provider(self) -> None:
        provider = FakeProvider()
        budget = GenerativeBudget(max_calls=0, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
        )
        source = _solid_rgb((40, 40), (50, 60, 70))
        mask = _rect_mask((40, 40), (5, 5, 20, 20))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 0)
        self.assertGreaterEqual(budget.fallbacks, 1)

    def test_restore_image_callable(self) -> None:
        provider = FakeProvider(fill_color=(11, 22, 33))
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=GenerativeBudget(max_calls=3, max_cost_usd=1.0),
            cache_dir=None,
            verify_ocr=False,
            low_freq_mae_threshold=-1.0,
            skip_flat_ring=False,
        )
        source = _solid_rgb((24, 24), (200, 200, 200))
        restored = engine.restore_image(source)
        self.assertEqual(restored.size, source.size)
        self.assertEqual(provider.calls, 1)

    def test_is_expected_safety_block_message(self) -> None:
        self.assertTrue(
            is_expected_safety_block_message(
                "Image generation blocked due to copyright/recitation."
            )
        )
        self.assertFalse(is_expected_safety_block_message("HTTP 503 unavailable"))


class ResolveAndCliTests(unittest.TestCase):
    def test_resolve_without_api_key_falls_back_locally(self) -> None:
        image = _solid_rgb((32, 32), (255, 255, 255))
        mask = _rect_mask((32, 32), (4, 4, 12, 12))
        options = ConversionOptions(
            input_path=Path("in.pdf"),
            output_path=Path("out.pptx"),
            report_path=Path("out.report.json"),
            inpaint_engine="generative-openai",
            generative_api_key="",
        )
        # Ensure env key is not accidentally present for this process assertion.
        engine, note = resolve_background_inpainting_engine(image, mask, options)
        self.assertNotIn("generative", engine.name)
        self.assertIn("no API key", note)

    def test_resolve_with_configured_provider(self) -> None:
        image = _solid_rgb((32, 32), (255, 255, 255))
        mask = _rect_mask((32, 32), (4, 4, 12, 12))
        options = ConversionOptions(
            input_path=Path("in.pdf"),
            output_path=Path("out.pptx"),
            report_path=Path("out.report.json"),
            inpaint_engine="generative-gemini",
            generative_api_key="test-key",
        )
        engine, note = resolve_background_inpainting_engine(image, mask, options)
        self.assertEqual(engine.name, "generative-gemini")
        self.assertIn("generative-gemini", note)
        self.assertTrue(isinstance(engine, GenerativeInpaintingEngine))

    def test_cli_accepts_generative_engines(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "in.pdf",
                "out.pptx",
                "--inpaint-engine",
                "generative-openai",
                "--generative-max-calls",
                "7",
                "--generative-max-cost-usd",
                "1.25",
                "--inpaint-mask-dilate-px",
                "3",
            ]
        )
        self.assertEqual(args.inpaint_engine, "generative-openai")
        self.assertEqual(args.generative_max_calls, 7)
        self.assertEqual(args.generative_max_cost_usd, 1.25)
        self.assertEqual(args.inpaint_mask_dilate_px, 3)

    def test_conversion_report_includes_generative_section(self) -> None:
        budget = GenerativeBudget(max_calls=3, max_cost_usd=1.0)
        budget.record_call(
            GenerativeCallRecord(
                provider="fake",
                model="fake-model",
                width=16,
                height=16,
                estimated_cost_usd=0.05,
                cache_hit=False,
                success=True,
                cost_source="usage",
            )
        )
        report = ConversionReport(
            input_path="in.pdf",
            output_path="out.pptx",
            pages=[
                PageResult(
                    page_number=1,
                    page_kind="scanned",
                    background_mode="overlay",
                    width_pt=100,
                    height_pt=100,
                    text_blocks=[],
                    quality_score=QualityScore(1.0, 1.0, 1.0, 1.0),
                    fallback_reason=None,
                )
            ],
            generative_inpaint=budget.to_dict(),
        )
        payload = report.to_dict()
        self.assertIn("generative_inpaint", payload)
        self.assertEqual(payload["generative_inpaint"]["calls"], 1)
        self.assertIn("cost_uncertain", payload["generative_inpaint"])


class ProviderUnitTests(unittest.TestCase):
    def test_openai_provider_builds_multipart_and_decodes_b64(self) -> None:
        edited = _solid_rgb((8, 8), (7, 8, 9))
        b64 = _png_b64(edited)

        def transport(*, body: bytes, content_type: str, api_key: str) -> dict:
            self.assertIn("multipart/form-data", content_type)
            self.assertTrue(api_key)
            self.assertIn(b"gpt-image-2", body)
            self.assertIn(b"image.png", body)
            self.assertIn(b"mask.png", body)
            # Prompt must not ask to remove watermarks/logos.
            self.assertNotIn(b"watermark", body.lower())
            return {
                "data": [{"b64_json": b64}],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "input_tokens_details": {"text_tokens": 20, "image_tokens": 80},
                },
            }

        provider = OpenAIImageEditProvider(api_key="sk-test", transport=transport)
        result = provider.edit(
            GenerativeEditRequest(
                image=_solid_rgb((8, 8), (1, 1, 1)),
                mask_image=_rect_mask((8, 8), (2, 2, 6, 6)),
                prompt=DEFAULT_TEXT_REMOVAL_PROMPT,
                model="gpt-image-2",
            )
        )
        self.assertEqual(result.image.size, (8, 8))
        self.assertEqual(result.provider_name, "openai")
        self.assertEqual(result.cost_source, "usage")
        self.assertFalse(result.cost_uncertain)
        self.assertGreater(result.estimated_cost_usd, 0.0)

    def test_gemini_provider_parses_steps_content_image(self) -> None:
        """Raw REST shape: steps[].content[] type=image (not output_image)."""

        edited = _solid_rgb((8, 8), (3, 4, 5))
        b64 = _png_b64(edited)
        captured: dict[str, object] = {}

        def transport(*, body: dict, api_key: str) -> dict:
            self.assertEqual(api_key, "gem-test")
            self.assertEqual(body["model"], "gemini-nano-banana-2.1")
            self.assertEqual(body["input"][0]["type"], "text")
            self.assertEqual(body["input"][1]["type"], "image")
            self.assertEqual(body["response_format"]["type"], "image")
            self.assertIn("aspect_ratio", body["response_format"])
            captured["aspect_ratio"] = body["response_format"]["aspect_ratio"]
            return {
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": "done"},
                            {"type": "image", "data": b64, "mime_type": "image/jpeg"},
                        ],
                    }
                ],
                "usage": {
                    "total_input_tokens": 1200,
                    "total_thought_tokens": 300,
                    "total_output_tokens": 1120,
                    "output_tokens_by_modality": [{"modality": "image", "tokens": 1120}],
                },
            }

        provider = GeminiImageEditProvider(api_key="gem-test", transport=transport)
        result = provider.edit(
            GenerativeEditRequest(
                image=_solid_rgb((8, 8), (9, 9, 9)),
                mask_image=_rect_mask((8, 8), (1, 1, 4, 4)),
                prompt=DEFAULT_TEXT_REMOVAL_PROMPT,
                model="gemini-nano-banana-2.1",
            )
        )
        self.assertEqual(result.image.size, (8, 8))
        self.assertEqual(result.provider_name, "gemini")
        self.assertEqual(captured["aspect_ratio"], "1:1")
        self.assertEqual(result.cost_source, "usage")
        self.assertAlmostEqual(result.estimated_cost_usd, 0.03765, places=5)

    def test_gemini_provider_still_accepts_sdk_output_image(self) -> None:
        edited = _solid_rgb((8, 8), (3, 4, 5))
        b64 = _png_b64(edited)

        def transport(*, body: dict, api_key: str) -> dict:
            del body, api_key
            return {"output_image": {"data": b64}}

        provider = GeminiImageEditProvider(api_key="gem-test", transport=transport)
        result = provider.edit(
            GenerativeEditRequest(
                image=_solid_rgb((8, 8), (9, 9, 9)),
                mask_image=_rect_mask((8, 8), (1, 1, 4, 4)),
                prompt="continue background",
                model="gemini-nano-banana-2.1",
            )
        )
        self.assertEqual(result.image.size, (8, 8))
        self.assertTrue(result.cost_uncertain)  # no usage → estimate

    def test_gemini_wide_crop_requests_matching_aspect_and_unpads(self) -> None:
        # Long thin crop; model "returns" a 4:1 canvas — must not stretch.
        crop_w, crop_h = 80, 20
        plan = plan_aspect_pad(crop_w, crop_h)
        model_canvas = Image.new("RGB", plan.padded_size, (1, 1, 1))
        left, top, right, bottom = plan.content_box
        for y in range(top, bottom):
            for x in range(left, right):
                model_canvas.putpixel((x, y), (50, 60, 70))
        b64 = _png_b64(model_canvas)
        captured: dict[str, object] = {}

        def transport(*, body: dict, api_key: str) -> dict:
            del api_key
            captured["aspect_ratio"] = body["response_format"]["aspect_ratio"]
            return {"steps": [{"type": "model_output", "content": [{"type": "image", "data": b64}]}]}

        provider = GeminiImageEditProvider(api_key="gem-test", transport=transport)
        result = provider.edit(
            GenerativeEditRequest(
                image=_solid_rgb((crop_w, crop_h), (9, 9, 9)),
                mask_image=_rect_mask((crop_w, crop_h), (10, 5, 70, 15)),
                prompt="continue background",
                model="gemini-nano-banana-2.1",
            )
        )
        self.assertEqual(captured["aspect_ratio"], plan.aspect_ratio)
        self.assertEqual(result.image.size, (crop_w, crop_h))
        # Content window should be the solid fill, not stretched garbage.
        arr = np.array(result.image)
        self.assertTrue(np.all(arr == (50, 60, 70)))

    def test_gemini_missing_image_marks_billed_uncertain(self) -> None:
        def transport(*, body: dict, api_key: str) -> dict:
            del body, api_key
            return {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "nope"}]}]}

        provider = GeminiImageEditProvider(api_key="gem-test", transport=transport)
        with self.assertRaises(GenerativeProviderError) as ctx:
            provider.edit(
                GenerativeEditRequest(
                    image=_solid_rgb((8, 8), (1, 1, 1)),
                    mask_image=_rect_mask((8, 8), (2, 2, 4, 4)),
                    prompt="x",
                    model="gemini-nano-banana-2.1",
                )
            )
        self.assertTrue(ctx.exception.may_have_been_billed)
        self.assertTrue(ctx.exception.cost_uncertain)

    def test_create_provider_registry(self) -> None:
        openai = create_provider("openai", api_key="x")
        gemini = create_provider("gemini", api_key="y")
        self.assertEqual(openai.name, "openai")
        self.assertEqual(gemini.name, "gemini")
        with self.assertRaises(ValueError):
            create_provider("nope")

    def test_openai_requires_key(self) -> None:
        provider = OpenAIImageEditProvider(api_key="")
        self.assertFalse(provider.is_configured())
        with self.assertRaises(GenerativeProviderError):
            provider.edit(
                GenerativeEditRequest(
                    image=_solid_rgb((4, 4), (0, 0, 0)),
                    mask_image=_rect_mask((4, 4), (1, 1, 2, 2)),
                    prompt="x",
                    model="gpt-image-2",
                )
            )


class MaskDilateOptionTests(unittest.TestCase):
    def test_default_mask_dilate_option(self) -> None:
        options = ConversionOptions(
            input_path=Path("in.pdf"),
            output_path=Path("out.pptx"),
            report_path=Path("out.report.json"),
        )
        self.assertEqual(options.inpaint_mask_dilate_px, 2)
        self.assertAlmostEqual(options.inpaint_mask_dilate_line_height_ratio, 0.18)
        self.assertEqual(options.generative_low_freq_mae_threshold, 4.0)
        self.assertTrue(options.generative_skip_flat_ring)
        self.assertEqual(options.generative_ring_texture_measure, "highpass")
        self.assertEqual(options.generative_ring_texture_blur_sigma, 5.0)
        self.assertEqual(options.generative_flat_ring_luma_std, 2.0)
        self.assertEqual(options.generative_textured_ring_luma_std, 4.0)
        self.assertEqual(options.generative_safety_block_retries, 1)

    def test_line_height_dilate_scales_and_merges(self) -> None:
        import pymupdf as fitz

        # Two nearby boxes: after line-height dilation they must merge into one
        # connected component (avoids solid_background 1→3 generative calls).
        size = (120, 40)
        base = Image.new("L", size, 0)
        for x0, x1 in ((10, 40), (48, 78)):
            for y in range(12, 28):
                for x in range(x0, x1):
                    base.putpixel((x, y), 255)
        blocks = [
            TextBlock(
                id="a",
                source="ocr",
                bbox=(10, 12, 40, 28),
                text="left",
                confidence=0.9,
                image_bbox=(10, 12, 40, 28),
            ),
            TextBlock(
                id="b",
                source="ocr",
                bbox=(48, 12, 78, 28),
                text="right",
                confidence=0.9,
                image_bbox=(48, 12, 78, 28),
            ),
        ]
        self.assertEqual(dilate_px_for_line_height(16, ratio=0.18, min_px=2, max_px=16, floor_px=2), 3)
        dilated = dilate_text_mask_by_line_height(
            base,
            blocks,
            fitz.Rect(0, 0, 120, 40),
            ratio=0.18,
            min_px=2,
            max_px=16,
            floor_px=2,
        )
        import cv2

        components, _, _, _ = cv2.connectedComponentsWithStats(
            (np.array(dilated) > 0).astype(np.uint8),
            8,
        )
        # background + one merged foreground component
        self.assertEqual(components, 2)


if __name__ == "__main__":
    unittest.main()
