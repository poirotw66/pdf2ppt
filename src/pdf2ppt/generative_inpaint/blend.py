"""Poisson / seamless blending and low-frequency quality gates for generative patches."""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

from ..inpainting_engines.compositing import _composite_lama_restoration_with_inference_mask

logger = logging.getLogger(__name__)

GENERATIVE_COMPOSITE_EDGE_FEATHER_PX = 3
DEFAULT_LOW_FREQ_BLUR_SIGMA = 12.0
DEFAULT_LOW_FREQ_MAE_THRESHOLD = 4.0
DEFAULT_CONTEXT_RING_PX = 8

# Ring texture measures used for flat-API-skip and low-freq gate scaling.
RingTextureMeasure = Literal["highpass", "plane_residual", "luma_std"]
RING_TEXTURE_MEASURES: tuple[str, ...] = ("highpass", "plane_residual", "luma_std")
DEFAULT_RING_TEXTURE_MEASURE: RingTextureMeasure = "highpass"
# Gaussian sigma for the high-pass residual (ring minus blur). Larger values keep
# more mid-frequency structure in the residual; 5px matches typical slide DPI.
DEFAULT_RING_TEXTURE_BLUR_SIGMA = 5.0
# Outside-ring texture score at or below this → treat as flat (skip generative API).
# Tuned for the default highpass measure: smooth gradients score ~0.2–0.5 while
# soft photo-like texture stays above this so Gemini is still attempted.
DEFAULT_FLAT_RING_TEXTURE_STD = 2.0
# Alias kept for callers/CLI that still use the historical "luma_std" name.
DEFAULT_FLAT_RING_LUMA_STD = DEFAULT_FLAT_RING_TEXTURE_STD
# When ring texture score reaches this, skip the low-freq gate (photo / textured pages).
# Kept relatively high so mild compression banding on gradient slides does not
# disable the gate; clear photo/noise rings still exceed it.
DEFAULT_TEXTURED_RING_TEXTURE_STD = 16.0
DEFAULT_TEXTURED_RING_LUMA_STD = DEFAULT_TEXTURED_RING_TEXTURE_STD
# Scale low-freq threshold as max(1, score / this) when between flat and textured.
DEFAULT_LOW_FREQ_STD_SCALE_REF = 2.0


def context_ring_mask(
    mask_array: np.ndarray,
    *,
    ring_px: int = DEFAULT_CONTEXT_RING_PX,
) -> np.ndarray:
    """Boolean ring just outside the mask (destination context for routing)."""

    binary = (mask_array > 0).astype(np.uint8)
    if not np.any(binary) or ring_px <= 0:
        return np.zeros(binary.shape, dtype=bool)
    kernel = np.ones((ring_px * 2 + 1, ring_px * 2 + 1), dtype=np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return (dilated > 0) & (binary == 0)


def _ring_gray_pixels(
    source_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    ring_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(gray, ring_mask, ring_values)`` for the outside context ring."""

    ring = context_ring_mask(mask_array, ring_px=ring_px)
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if not np.any(ring):
        return gray, ring, np.empty(0, dtype=np.float32)
    return gray, ring, gray[ring]


def context_ring_luma_std(
    source_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    ring_px: int = DEFAULT_CONTEXT_RING_PX,
) -> float:
    """Luma standard deviation on the outside ring around the mask."""

    _gray, _ring, values = _ring_gray_pixels(source_rgb, mask_array, ring_px=ring_px)
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _fill_mask_from_ring_surface(
    gray: np.ndarray,
    mask_array: np.ndarray,
    ring: np.ndarray,
) -> np.ndarray:
    """Replace masked pixels with a low-order surface fit from the outside ring.

    Uses a quadratic (6-parameter) fit when enough ring samples exist, otherwise
    a plane, otherwise the ring mean. This prevents glyph edges under the mask
    from bleeding into a Gaussian blur while preserving smooth linear and radial
    gradients (a constant fill would create a false plateau).
    """

    filled = gray.copy()
    binary_mask = mask_array > 0
    if not np.any(binary_mask) or not np.any(ring):
        return filled
    ys, xs = np.where(ring)
    z = gray[ys, xs].astype(np.float64)
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    my, mx = np.where(binary_mask)
    mx_f = mx.astype(np.float64)
    my_f = my.astype(np.float64)
    if ys.size >= 6:
        design = np.column_stack([x * x, y * y, x * y, x, y, np.ones(x.size, dtype=np.float64)])
        coefficients, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
        filled[my, mx] = (
            coefficients[0] * mx_f * mx_f
            + coefficients[1] * my_f * my_f
            + coefficients[2] * mx_f * my_f
            + coefficients[3] * mx_f
            + coefficients[4] * my_f
            + coefficients[5]
        ).astype(np.float32)
        return filled
    if ys.size >= 3:
        design = np.column_stack([x, y, np.ones(x.size, dtype=np.float64)])
        coefficients, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
        filled[my, mx] = (
            coefficients[0] * mx_f + coefficients[1] * my_f + coefficients[2]
        ).astype(np.float32)
        return filled
    filled[binary_mask] = float(np.mean(z))
    return filled


def context_ring_highpass_std(
    source_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    ring_px: int = DEFAULT_CONTEXT_RING_PX,
    blur_sigma: float = DEFAULT_RING_TEXTURE_BLUR_SIGMA,
) -> float:
    """Std of high-frequency luma on the outside ring (ring minus Gaussian blur).

    Smooth linear/radial gradients have near-zero high-pass energy even when raw
    luma std is large across the ring. Photo / noisy texture keeps substantial
    residual energy.

    Masked interior pixels are filled from a low-order surface fit on the ring
    before blurring so glyph edges under the mask do not bleed into the residual.
    """

    gray, ring, values = _ring_gray_pixels(source_rgb, mask_array, ring_px=ring_px)
    if not np.any(ring):
        return 0.0
    if blur_sigma <= 0:
        return float(np.std(values))
    filled = _fill_mask_from_ring_surface(gray, mask_array, ring)
    blurred = np.asarray(
        cv2.GaussianBlur(filled, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma),
        dtype=np.float32,
    )
    residual = filled - blurred
    return float(np.std(residual[ring]))


def context_ring_plane_residual_std(
    source_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    ring_px: int = DEFAULT_CONTEXT_RING_PX,
) -> float:
    """Std of residual after fitting a plane to ring luma samples.

    Linear gradients fit well (low residual). Radial gradients and textured
    backgrounds leave larger residuals. Prefer ``highpass`` when radial slides
    must also classify as flat.
    """

    gray, ring, _values = _ring_gray_pixels(source_rgb, mask_array, ring_px=ring_px)
    if not np.any(ring):
        return 0.0
    ys, xs = np.where(ring)
    if ys.size < 3:
        return float(np.std(gray[ring]))
    z = gray[ys, xs].astype(np.float64)
    design = np.column_stack(
        [xs.astype(np.float64), ys.astype(np.float64), np.ones(ys.size, dtype=np.float64)]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - design @ coefficients
    return float(np.std(residual))


def normalize_ring_texture_measure(measure: str) -> RingTextureMeasure:
    """Validate and normalize a ring texture measure name."""

    resolved = str(measure).strip().lower().replace("-", "_")
    if resolved not in RING_TEXTURE_MEASURES:
        allowed = ", ".join(RING_TEXTURE_MEASURES)
        raise ValueError(f"Unknown ring texture measure {measure!r}; expected one of: {allowed}")
    return resolved  # type: ignore[return-value]


def context_ring_texture_score(
    source_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    measure: str = DEFAULT_RING_TEXTURE_MEASURE,
    ring_px: int = DEFAULT_CONTEXT_RING_PX,
    blur_sigma: float = DEFAULT_RING_TEXTURE_BLUR_SIGMA,
) -> float:
    """Texture score on the outside ring using the configured measure."""

    resolved = normalize_ring_texture_measure(measure)
    if resolved == "luma_std":
        return context_ring_luma_std(source_rgb, mask_array, ring_px=ring_px)
    if resolved == "plane_residual":
        return context_ring_plane_residual_std(source_rgb, mask_array, ring_px=ring_px)
    return context_ring_highpass_std(
        source_rgb,
        mask_array,
        ring_px=ring_px,
        blur_sigma=blur_sigma,
    )


def effective_low_freq_mae_threshold(
    base_threshold: float,
    ring_texture_score: float,
    *,
    textured_std: float = DEFAULT_TEXTURED_RING_TEXTURE_STD,
    scale_ref_std: float = DEFAULT_LOW_FREQ_STD_SCALE_REF,
) -> float | None:
    """Return the effective low-freq MAE threshold, or ``None`` to skip the gate.

    Flat/near-flat surroundings keep the base threshold. Highly textured rings
    (photo backgrounds) disable the gate so generative fills are not rejected for
    differing high-frequency content. Mid-range scores scale the threshold up.
    """

    if base_threshold < 0:
        return None
    if ring_texture_score >= textured_std:
        return None
    if scale_ref_std <= 0:
        return base_threshold
    scale = max(1.0, ring_texture_score / scale_ref_std)
    return float(base_threshold) * scale


def low_frequency_masked_mae(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    blur_sigma: float = DEFAULT_LOW_FREQ_BLUR_SIGMA,
) -> float:
    """Mean absolute RGB error inside ``mask`` after heavy Gaussian blur (0–255 scale)."""

    binary = mask_array > 0
    if not np.any(binary):
        return 0.0
    left: np.ndarray = left_rgb.astype(np.float32)
    right: np.ndarray = right_rgb.astype(np.float32)
    if blur_sigma > 0:
        left = np.asarray(
            cv2.GaussianBlur(left, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma),
            dtype=np.float32,
        )
        right = np.asarray(
            cv2.GaussianBlur(right, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma),
            dtype=np.float32,
        )
    diff = np.abs(left - right)
    return float(np.mean(diff[binary]))


def seamless_clone_masked_edit(
    source_rgb: np.ndarray,
    edited_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    clone_mode: int | None = None,
    edge_feather_px: int = GENERATIVE_COMPOSITE_EDGE_FEATHER_PX,
) -> np.ndarray:
    """Poisson-blend ``edited_rgb`` into ``source_rgb`` through ``mask_array``.

    Defaults to ``NORMAL_CLONE``. ``MIXED_CLONE`` must not be auto-selected from
    the original (text-filled) crop: it keeps the stronger of source/destination
    gradients and pulls glyph edges back into a clean generative fill.
    """

    if source_rgb.shape != edited_rgb.shape:
        raise ValueError(
            f"source/edited shape mismatch: {source_rgb.shape} vs {edited_rgb.shape}"
        )
    binary = (mask_array > 0).astype(np.uint8)
    if not np.any(binary):
        return source_rgb.copy()

    if clone_mode is None:
        clone_mode = cv2.NORMAL_CLONE

    # seamlessClone is unstable when the mask touches the image border; inset by
    # 1px when possible so the solver has a valid exterior ring.
    height, width = binary.shape[:2]
    inset = binary.copy()
    inset[0, :] = 0
    inset[-1, :] = 0
    inset[:, 0] = 0
    inset[:, -1] = 0
    if not np.any(inset):
        return _composite_lama_restoration_with_inference_mask(
            source_rgb,
            edited_rgb,
            mask_array,
            edge_feather_px=edge_feather_px,
        )

    ys, xs = np.where(inset > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    center = (int((x0 + x1) // 2), int((y0 + y1) // 2))
    # Require a non-degenerate bounding box so OpenCV does not raise.
    if x1 <= x0 or y1 <= y0:
        return _composite_lama_restoration_with_inference_mask(
            source_rgb,
            edited_rgb,
            mask_array,
            edge_feather_px=edge_feather_px,
        )

    src_bgr = cv2.cvtColor(edited_rgb, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    mask_u8 = (inset * 255).astype(np.uint8)
    try:
        cloned_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask_u8, center, clone_mode)
    except cv2.error as error:
        logger.debug("seamlessClone failed (%s); using feather composite", error)
        return _composite_lama_restoration_with_inference_mask(
            source_rgb,
            edited_rgb,
            mask_array,
            edge_feather_px=edge_feather_px,
        )
    cloned_rgb = cv2.cvtColor(cloned_bgr, cv2.COLOR_BGR2RGB)
    # Keep a hard lock outside the original (non-inset) mask.
    return _composite_lama_restoration_with_inference_mask(
        source_rgb,
        cloned_rgb,
        mask_array,
        edge_feather_px=edge_feather_px,
    )


__all__ = [
    "DEFAULT_CONTEXT_RING_PX",
    "DEFAULT_FLAT_RING_LUMA_STD",
    "DEFAULT_FLAT_RING_TEXTURE_STD",
    "DEFAULT_LOW_FREQ_BLUR_SIGMA",
    "DEFAULT_LOW_FREQ_MAE_THRESHOLD",
    "DEFAULT_LOW_FREQ_STD_SCALE_REF",
    "DEFAULT_RING_TEXTURE_BLUR_SIGMA",
    "DEFAULT_RING_TEXTURE_MEASURE",
    "DEFAULT_TEXTURED_RING_LUMA_STD",
    "DEFAULT_TEXTURED_RING_TEXTURE_STD",
    "GENERATIVE_COMPOSITE_EDGE_FEATHER_PX",
    "RING_TEXTURE_MEASURES",
    "RingTextureMeasure",
    "context_ring_highpass_std",
    "context_ring_luma_std",
    "context_ring_mask",
    "context_ring_plane_residual_std",
    "context_ring_texture_score",
    "effective_low_freq_mae_threshold",
    "low_frequency_masked_mae",
    "normalize_ring_texture_measure",
    "seamless_clone_masked_edit",
]
