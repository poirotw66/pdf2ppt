"""Opt-in generative image-model backend for text-region inpainting.

This package implements Phase 4 of ``docs/optimization-roadmap.md``: cloud image
edit models (OpenAI GPT Image / Google Gemini Nano Banana) as an explicit
opt-in alternative to local OpenCV / LaMa inpainting.

Design invariants:
- No API key => existing local behaviour is unchanged (engines are never selected
  by ``auto``).
- Only masked pixels may change; results are hard-composited back onto the
  source image.
- API errors, budget exhaustion, and OCR residual-text failures fall back to
  local ``opencv-fast``.
- Results are cached by (provider, model, image, mask, prompt) hash.
"""

from __future__ import annotations

from .blend import (
    DEFAULT_FLAT_RING_LUMA_STD,
    DEFAULT_FLAT_RING_TEXTURE_STD,
    DEFAULT_LOW_FREQ_MAE_THRESHOLD,
    DEFAULT_RING_TEXTURE_BLUR_SIGMA,
    DEFAULT_RING_TEXTURE_MEASURE,
    DEFAULT_TEXTURED_RING_LUMA_STD,
    DEFAULT_TEXTURED_RING_TEXTURE_STD,
    RING_TEXTURE_MEASURES,
    context_ring_highpass_std,
    context_ring_luma_std,
    context_ring_plane_residual_std,
    context_ring_texture_score,
    effective_low_freq_mae_threshold,
    low_frequency_masked_mae,
    normalize_ring_texture_measure,
    seamless_clone_masked_edit,
)
from .budget import GenerativeBudget, GenerativeCallRecord
from .cache import GenerativeInpaintCache, compute_generative_cache_key
from .engine import (
    DEFAULT_SAFETY_BLOCK_RETRIES,
    GENERATIVE_INPAINT_ENGINES,
    GenerativeInpaintingEngine,
    composite_masked_edit,
    provider_name_for_engine,
    uses_generative_inpaint_engine,
)
from .providers import (
    DEFAULT_IMAGE_RESTORE_PROMPT,
    DEFAULT_TEXT_REMOVAL_PROMPT,
    GEMINI_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    GeminiImageEditProvider,
    GenerativeEditRequest,
    GenerativeEditResult,
    GenerativeProviderError,
    OpenAIImageEditProvider,
    create_provider,
    is_expected_safety_block_message,
)
from .safety import (
    build_ocr_residual_checker,
    residual_text_overlap_ratio,
)

__all__ = [
    "DEFAULT_FLAT_RING_LUMA_STD",
    "DEFAULT_FLAT_RING_TEXTURE_STD",
    "DEFAULT_IMAGE_RESTORE_PROMPT",
    "DEFAULT_LOW_FREQ_MAE_THRESHOLD",
    "DEFAULT_RING_TEXTURE_BLUR_SIGMA",
    "DEFAULT_RING_TEXTURE_MEASURE",
    "DEFAULT_SAFETY_BLOCK_RETRIES",
    "DEFAULT_TEXT_REMOVAL_PROMPT",
    "DEFAULT_TEXTURED_RING_LUMA_STD",
    "DEFAULT_TEXTURED_RING_TEXTURE_STD",
    "GEMINI_DEFAULT_MODEL",
    "GENERATIVE_INPAINT_ENGINES",
    "OPENAI_DEFAULT_MODEL",
    "RING_TEXTURE_MEASURES",
    "GenerativeBudget",
    "GenerativeCallRecord",
    "GenerativeEditRequest",
    "GenerativeEditResult",
    "GenerativeInpaintCache",
    "GenerativeInpaintingEngine",
    "GenerativeProviderError",
    "GeminiImageEditProvider",
    "OpenAIImageEditProvider",
    "build_ocr_residual_checker",
    "composite_masked_edit",
    "compute_generative_cache_key",
    "context_ring_highpass_std",
    "context_ring_luma_std",
    "context_ring_plane_residual_std",
    "context_ring_texture_score",
    "create_provider",
    "effective_low_freq_mae_threshold",
    "is_expected_safety_block_message",
    "low_frequency_masked_mae",
    "normalize_ring_texture_measure",
    "provider_name_for_engine",
    "residual_text_overlap_ratio",
    "seamless_clone_masked_edit",
    "uses_generative_inpaint_engine",
]
