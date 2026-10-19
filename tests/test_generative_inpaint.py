"""Tests for the opt-in generative inpaint backend (providers mocked; no network)."""

from __future__ import annotations

import base64
import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pdf2ppt.cli import build_parser
from pdf2ppt.core import ConversionOptions
from pdf2ppt.generative_inpaint import (
    DEFAULT_TEXT_REMOVAL_PROMPT,
    GeminiImageEditProvider,
    GenerativeBudget,
    GenerativeEditRequest,
    GenerativeEditResult,
    GenerativeInpaintingEngine,
    GenerativeProviderError,
    OpenAIImageEditProvider,
    composite_masked_edit,
    compute_generative_cache_key,
    create_provider,
    uses_generative_inpaint_engine,
)
from pdf2ppt.generative_inpaint.cache import GenerativeInpaintCache
from pdf2ppt.generative_inpaint.safety import assert_unmasked_pixels_unchanged
from pdf2ppt.inpainting_overlay import resolve_background_inpainting_engine
from pdf2ppt.models import ConversionReport, PageResult, QualityScore


def _solid_rgb(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def _rect_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    x0, y0, x1, y1 = box
    for y in range(y0, y1):
        for x in range(x0, x1):
            mask.putpixel((x, y), 255)
    return mask


class FakeProvider:
    name = "fake"
    default_model = "fake-model"

    def __init__(self, *, fill_color: tuple[int, int, int] = (10, 20, 30), fail: bool = False) -> None:
        self.fill_color = fill_color
        self.fail = fail
        self.calls = 0
        self._configured = True

    def is_configured(self) -> bool:
        return self._configured

    def estimate_cost_usd(self, *, model: str, width: int, height: int) -> float:
        del model, width, height
        return 0.05

    def edit(self, request: GenerativeEditRequest) -> GenerativeEditResult:
        self.calls += 1
        if self.fail:
            raise GenerativeProviderError("simulated provider failure")
        # Paint the whole crop a solid color; compositor must keep unmasked pixels.
        edited = Image.new("RGB", request.image.size, self.fill_color)
        return GenerativeEditResult(
            image=edited,
            model=self.default_model,
            provider_name=self.name,
            estimated_cost_usd=0.05,
            raw_response_note="fake",
        )


class GenerativeHelpersTests(unittest.TestCase):
    def test_uses_generative_engine_names(self) -> None:
        self.assertTrue(uses_generative_inpaint_engine("generative-openai"))
        self.assertTrue(uses_generative_inpaint_engine("generative-gemini"))
        self.assertFalse(uses_generative_inpaint_engine("opencv-fast"))
        self.assertFalse(uses_generative_inpaint_engine("auto"))

    def test_composite_preserves_unmasked_pixels(self) -> None:
        source = _solid_rgb((32, 32), (200, 100, 50))
        edited = _solid_rgb((32, 32), (1, 2, 3))
        mask = _rect_mask((32, 32), (8, 8, 24, 24))
        result = composite_masked_edit(source, edited, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        source_arr = np.array(source)
        result_arr = np.array(result)
        mask_arr = np.array(mask) > 0
        self.assertFalse(np.array_equal(source_arr[mask_arr], result_arr[mask_arr]))

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


class GenerativeEngineTests(unittest.TestCase):
    def test_engine_inpaints_only_mask_and_records_budget(self) -> None:
        provider = FakeProvider(fill_color=(9, 9, 9))
        budget = GenerativeBudget(max_calls=5, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
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
        )
        source = _solid_rgb((48, 48), (120, 130, 140))
        mask = _rect_mask((48, 48), (10, 10, 30, 30))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertEqual(provider.calls, 1)
        self.assertGreaterEqual(engine.budget.fallbacks, 1)

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
        )
        source = _solid_rgb((48, 48), (90, 90, 90))
        mask = _rect_mask((48, 48), (12, 12, 36, 36))
        result = engine.inpaint(source, mask)
        assert_unmasked_pixels_unchanged(source, result, mask)
        self.assertGreaterEqual(engine.budget.fallbacks, 1)

    def test_budget_exhaustion_skips_provider(self) -> None:
        provider = FakeProvider()
        budget = GenerativeBudget(max_calls=0, max_cost_usd=1.0)
        engine = GenerativeInpaintingEngine(
            provider=provider,
            budget=budget,
            cache_dir=None,
            verify_ocr=False,
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
        )
        source = _solid_rgb((24, 24), (200, 200, 200))
        restored = engine.restore_image(source)
        self.assertEqual(restored.size, source.size)
        self.assertEqual(provider.calls, 1)


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
            ]
        )
        self.assertEqual(args.inpaint_engine, "generative-openai")
        self.assertEqual(args.generative_max_calls, 7)
        self.assertEqual(args.generative_max_cost_usd, 1.25)

    def test_conversion_report_includes_generative_section(self) -> None:
        budget = GenerativeBudget(max_calls=3, max_cost_usd=1.0)
        budget.record_call(
            __import__("pdf2ppt.generative_inpaint.budget", fromlist=["GenerativeCallRecord"]).GenerativeCallRecord(
                provider="fake",
                model="fake-model",
                width=16,
                height=16,
                estimated_cost_usd=0.05,
                cache_hit=False,
                success=True,
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


class ProviderUnitTests(unittest.TestCase):
    def test_openai_provider_builds_multipart_and_decodes_b64(self) -> None:
        edited = _solid_rgb((8, 8), (7, 8, 9))
        buffer = io.BytesIO()
        edited.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        def transport(*, body: bytes, content_type: str, api_key: str) -> dict:
            self.assertIn("multipart/form-data", content_type)
            self.assertTrue(api_key)
            self.assertIn(b"gpt-image-2", body)
            self.assertIn(b"image.png", body)
            self.assertIn(b"mask.png", body)
            return {"data": [{"b64_json": b64}], "usage": {"total_tokens": 1}}

        provider = OpenAIImageEditProvider(api_key="sk-test", transport=transport)
        result = provider.edit(
            GenerativeEditRequest(
                image=_solid_rgb((8, 8), (1, 1, 1)),
                mask_image=_rect_mask((8, 8), (2, 2, 6, 6)),
                prompt="remove text",
                model="gpt-image-2",
            )
        )
        self.assertEqual(result.image.size, (8, 8))
        self.assertEqual(result.provider_name, "openai")

    def test_gemini_provider_posts_json_and_decodes_output_image(self) -> None:
        edited = _solid_rgb((8, 8), (3, 4, 5))
        buffer = io.BytesIO()
        edited.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        def transport(*, body: dict, api_key: str) -> dict:
            self.assertEqual(api_key, "gem-test")
            self.assertEqual(body["model"], "gemini-nano-banana-2.1")
            self.assertEqual(body["input"][0]["type"], "text")
            self.assertEqual(body["input"][1]["type"], "image")
            return {"output_image": {"data": b64}}

        provider = GeminiImageEditProvider(api_key="gem-test", transport=transport)
        result = provider.edit(
            GenerativeEditRequest(
                image=_solid_rgb((8, 8), (9, 9, 9)),
                mask_image=_rect_mask((8, 8), (1, 1, 4, 4)),
                prompt="remove text",
                model="gemini-nano-banana-2.1",
            )
        )
        self.assertEqual(result.image.size, (8, 8))
        self.assertEqual(result.provider_name, "gemini")

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


if __name__ == "__main__":
    unittest.main()
