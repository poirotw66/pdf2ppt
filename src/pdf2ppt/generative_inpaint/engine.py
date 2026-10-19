"""Generative cloud inpainting engine implementing BackgroundInpaintingEngine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..inpainting_engines.base import (
    DEFAULT_LAMA_PATCH_CONTEXT_PADDING_PX,
    DEFAULT_LAMA_PATCH_MERGE_GAP_PX,
    BackgroundInpaintingEngine,
    BackgroundInpaintingError,
)
from ..inpainting_engines.compositing import _composite_lama_restoration_with_inference_mask
from ..inpainting_engines.opencv_fast import OpenCvFastInpaintingEngine
from ..inpainting_engines.patching import (
    _build_lama_patch_groups,
    _prepare_lama_working_mask,
    _should_use_lama_patch_inpaint,
)
from .budget import GenerativeBudget, GenerativeCallRecord
from .cache import GenerativeInpaintCache, compute_generative_cache_key
from .providers import (
    DEFAULT_IMAGE_RESTORE_PROMPT,
    DEFAULT_TEXT_REMOVAL_PROMPT,
    GenerativeEditRequest,
    GenerativeImageEditProvider,
    GenerativeProviderError,
    create_provider,
)
from .safety import ResidualChecker, assert_unmasked_pixels_unchanged

logger = logging.getLogger(__name__)

GENERATIVE_INPAINT_ENGINES = frozenset({"generative-openai", "generative-gemini"})
GENERATIVE_ENGINE_PROVIDER = {
    "generative-openai": "openai",
    "generative-gemini": "gemini",
}


def uses_generative_inpaint_engine(engine: str) -> bool:
    return engine in GENERATIVE_INPAINT_ENGINES


def provider_name_for_engine(engine: str) -> str | None:
    return GENERATIVE_ENGINE_PROVIDER.get(engine)


def composite_masked_edit(
    source: Image.Image,
    edited: Image.Image,
    mask_image: Image.Image,
) -> Image.Image:
    """Force unmasked pixels to stay identical to ``source`` after a generative edit."""

    source_rgb = np.array(source.convert("RGB"), dtype=np.uint8)
    edited_rgb = np.array(edited.convert("RGB"), dtype=np.uint8)
    if edited_rgb.shape != source_rgb.shape:
        edited = edited.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
        edited_rgb = np.array(edited, dtype=np.uint8)
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    if mask_array.shape[:2] != source_rgb.shape[:2]:
        mask_image = mask_image.convert("L").resize(source.size, Image.Resampling.NEAREST)
        mask_array = np.array(mask_image, dtype=np.uint8)
    blended = _composite_lama_restoration_with_inference_mask(
        source_rgb,
        edited_rgb,
        mask_array,
        edge_feather_px=2,
    )
    result = Image.fromarray(blended, mode="RGB")
    assert_unmasked_pixels_unchanged(source, result, mask_image)
    return result


class GenerativeInpaintingEngine(BackgroundInpaintingEngine):
    """Opt-in cloud generative inpaint engine with budget, cache, and OCR safety."""

    name = "generative"

    def __init__(
        self,
        *,
        provider: GenerativeImageEditProvider | str,
        model: str | None = None,
        api_key: str | None = None,
        max_calls: int = 50,
        max_cost_usd: float = 5.0,
        cache_dir: Path | None = None,
        timeout_s: float = 120.0,
        verify_ocr: bool = True,
        residual_checker: ResidualChecker | None = None,
        budget: GenerativeBudget | None = None,
        prompt: str | None = None,
        transport: Any | None = None,
    ) -> None:
        resolved_provider: GenerativeImageEditProvider
        if isinstance(provider, str):
            resolved_provider = create_provider(
                provider,
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
                transport=transport,
            )
            self.name = f"generative-{provider}"
        else:
            resolved_provider = provider
            self.name = f"generative-{provider.name}"
        self._provider = resolved_provider
        configured = getattr(self._provider, "configured_model", None)
        resolved_model = model or configured or self._provider.default_model
        self._model: str = str(resolved_model)
        self._cache = GenerativeInpaintCache(cache_dir)
        self._budget = budget or GenerativeBudget(max_calls=max_calls, max_cost_usd=max_cost_usd)
        self._verify_ocr = verify_ocr
        self._residual_checker = residual_checker
        self._prompt = prompt or DEFAULT_TEXT_REMOVAL_PROMPT
        self._last_debug_note: str | None = None
        self._last_debug_images: dict[str, Image.Image] = {}
        self._local_fallback = OpenCvFastInpaintingEngine()

    @property
    def budget(self) -> GenerativeBudget:
        return self._budget

    @property
    def last_debug_note(self) -> str | None:
        return self._last_debug_note

    @property
    def last_debug_images(self) -> dict[str, Image.Image]:
        return self._last_debug_images

    def is_configured(self) -> bool:
        return self._provider.is_configured()

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        if not self.is_configured():
            raise BackgroundInpaintingError(
                f"Generative provider {self._provider.name!r} is not configured "
                "(missing API key). Falling back to local inpainting."
            )

        source = page_image.convert("RGB")
        mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
        working_mask = _prepare_lama_working_mask(mask_array)
        source_rgb = np.array(source, dtype=np.uint8)

        if _should_use_lama_patch_inpaint(working_mask):
            result_rgb, note = self._inpaint_by_patches(source_rgb, working_mask)
        else:
            result_rgb, note = self._inpaint_full_or_single(source_rgb, working_mask)

        result = Image.fromarray(result_rgb, mode="RGB")
        assert_unmasked_pixels_unchanged(source, result, Image.fromarray(working_mask, mode="L"))
        self._last_debug_note = note
        self._last_debug_images = {
            "generative_working_mask": Image.fromarray(working_mask, mode="L"),
        }
        return result

    def restore_image(self, image: Image.Image, *, prompt: str | None = None) -> Image.Image:
        """Optional full-image quality restore (callable for later benchmarking).

        This is intentionally separate from text-mask inpainting. It still requires
        a configured provider and respects the same call/cost budget.
        """

        if not self._provider.is_configured():
            raise BackgroundInpaintingError(
                f"Generative provider {self._provider.name!r} is not configured."
            )
        estimate = self._provider.estimate_cost_usd(
            model=self._model,
            width=image.width,
            height=image.height,
        )
        if not self._budget.can_afford(estimate):
            raise BackgroundInpaintingError("Generative restore blocked by call/cost budget.")

        # Empty mask means "edit nothing" for masked APIs; for restore we send a full mask.
        full_mask = Image.new("L", image.size, 255)
        request = GenerativeEditRequest(
            image=image.convert("RGB"),
            mask_image=full_mask,
            prompt=prompt or DEFAULT_IMAGE_RESTORE_PROMPT,
            model=self._model,
        )
        try:
            edit_result = self._provider.edit(request)
        except GenerativeProviderError as error:
            self._budget.record_call(
                GenerativeCallRecord(
                    provider=self._provider.name,
                    model=self._model,
                    width=image.width,
                    height=image.height,
                    estimated_cost_usd=0.0,
                    cache_hit=False,
                    success=False,
                    note=str(error),
                )
            )
            raise BackgroundInpaintingError(str(error)) from error

        self._budget.record_call(
            GenerativeCallRecord(
                provider=edit_result.provider_name,
                model=edit_result.model,
                width=image.width,
                height=image.height,
                estimated_cost_usd=edit_result.estimated_cost_usd,
                cache_hit=False,
                success=True,
                note="image-restore",
            )
        )
        restored = edit_result.image.convert("RGB")
        if restored.size != image.size:
            restored = restored.resize(image.size, Image.Resampling.LANCZOS)
        return restored

    def _inpaint_full_or_single(
        self,
        source_rgb: np.ndarray,
        working_mask: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        crop_mask = working_mask
        restored, path = self._inpaint_crop(source_rgb, crop_mask)
        return restored, f"generative full-page path={path}"

    def _inpaint_by_patches(
        self,
        source_rgb: np.ndarray,
        working_mask: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        patch_groups = _build_lama_patch_groups(
            working_mask,
            context_padding_px=DEFAULT_LAMA_PATCH_CONTEXT_PADDING_PX,
            merge_gap_px=DEFAULT_LAMA_PATCH_MERGE_GAP_PX,
        )
        result = source_rgb.copy()
        generative_count = 0
        fallback_count = 0
        for x0, y0, x1, y1, group_mask in patch_groups:
            crop_source = source_rgb[y0:y1, x0:x1]
            crop_mask = group_mask[y0:y1, x0:x1]
            if not np.any(crop_mask):
                continue
            crop_working_mask = _prepare_lama_working_mask(crop_mask)
            restored_crop, path = self._inpaint_crop(crop_source, crop_working_mask)
            if path.startswith("generative"):
                generative_count += 1
            else:
                fallback_count += 1
            result[y0:y1, x0:x1] = _composite_lama_restoration_with_inference_mask(
                result[y0:y1, x0:x1],
                restored_crop,
                crop_working_mask,
                edge_feather_px=2,
            )
        note = (
            f"generative patch-mode groups={len(patch_groups)} "
            f"generative={generative_count} fallback={fallback_count} "
            f"budget_calls={self._budget.calls}/{self._budget.max_calls} "
            f"est_cost_usd={self._budget.estimated_cost_usd:.4f}"
        )
        return result, note

    def _inpaint_crop(
        self,
        crop_source: np.ndarray,
        crop_mask: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        crop_image = Image.fromarray(crop_source, mode="RGB")
        mask_image = Image.fromarray(crop_mask, mode="L")
        estimate = self._provider.estimate_cost_usd(
            model=self._model,
            width=crop_image.width,
            height=crop_image.height,
        )
        cache_key = compute_generative_cache_key(
            provider=self._provider.name,
            model=self._model,
            image=crop_image,
            mask_image=mask_image,
            prompt=self._prompt,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._budget.record_cache_hit(
                GenerativeCallRecord(
                    provider=self._provider.name,
                    model=self._model,
                    width=crop_image.width,
                    height=crop_image.height,
                    estimated_cost_usd=0.0,
                    cache_hit=True,
                    success=True,
                )
            )
            composited = composite_masked_edit(crop_image, cached, mask_image)
            if self._passes_safety(composited, mask_image):
                return np.array(composited, dtype=np.uint8), "generative-cache"
            self._budget.record_fallback("cache rejected by OCR safety")

        if not self._budget.can_afford(estimate):
            self._budget.record_fallback("budget exhausted")
            return self._local_inpaint(crop_source, crop_mask), "local-budget"

        request = GenerativeEditRequest(
            image=crop_image,
            mask_image=mask_image,
            prompt=self._prompt,
            model=self._model,
        )
        try:
            edit_result = self._provider.edit(request)
        except GenerativeProviderError as error:
            logger.warning("Generative provider failed; using local fallback: %s", error)
            self._budget.record_call(
                GenerativeCallRecord(
                    provider=self._provider.name,
                    model=self._model,
                    width=crop_image.width,
                    height=crop_image.height,
                    estimated_cost_usd=0.0,
                    cache_hit=False,
                    success=False,
                    note=str(error),
                )
            )
            self._budget.record_fallback(str(error))
            return self._local_inpaint(crop_source, crop_mask), "local-api-error"

        self._budget.record_call(
            GenerativeCallRecord(
                provider=edit_result.provider_name,
                model=edit_result.model,
                width=crop_image.width,
                height=crop_image.height,
                estimated_cost_usd=edit_result.estimated_cost_usd,
                cache_hit=False,
                success=True,
                note=edit_result.raw_response_note,
            )
        )
        composited = composite_masked_edit(crop_image, edit_result.image, mask_image)
        if not self._passes_safety(composited, mask_image):
            self._budget.record_fallback("OCR safety rejected generative patch")
            return self._local_inpaint(crop_source, crop_mask), "local-ocr-safety"

        self._cache.put(cache_key, edit_result.image.convert("RGB"))
        return np.array(composited, dtype=np.uint8), "generative"

    def _passes_safety(self, image: Image.Image, mask_image: Image.Image) -> bool:
        if not self._verify_ocr:
            return True
        if self._residual_checker is None:
            # No OCR wired for this run; skip rather than blocking generative use.
            return True
        return self._residual_checker(image, mask_image)

    def _local_inpaint(self, crop_source: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        crop_page = Image.fromarray(crop_source, mode="RGB")
        crop_mask_image = Image.fromarray(crop_mask, mode="L")
        return np.array(self._local_fallback.inpaint(crop_page, crop_mask_image), dtype=np.uint8)


__all__ = [
    "GENERATIVE_INPAINT_ENGINES",
    "GenerativeInpaintingEngine",
    "composite_masked_edit",
    "provider_name_for_engine",
    "uses_generative_inpaint_engine",
]
