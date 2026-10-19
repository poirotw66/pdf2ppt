"""Poisson / seamless blending and low-frequency quality gates for generative patches."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..inpainting_engines.compositing import _composite_lama_restoration_with_inference_mask

logger = logging.getLogger(__name__)

GENERATIVE_COMPOSITE_EDGE_FEATHER_PX = 3
DEFAULT_LOW_FREQ_BLUR_SIGMA = 12.0
DEFAULT_LOW_FREQ_MAE_THRESHOLD = 4.0
# Surrounding-ring luma std at or below this → treat as flat (skip generative API).
DEFAULT_FLAT_RING_LUMA_STD = 4.0
DEFAULT_CONTEXT_RING_PX = 8
# When ring std reaches this, skip the low-freq gate (photo / textured pages).
DEFAULT_TEXTURED_RING_LUMA_STD = 16.0
# Scale low-freq threshold as max(1, ring_std / this) when between flat and textured.
DEFAULT_LOW_FREQ_STD_SCALE_REF = 4.0


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


def context_ring_luma_std(
    source_rgb: np.ndarray,
    mask_array: np.ndarray,
    *,
    ring_px: int = DEFAULT_CONTEXT_RING_PX,
) -> float:
    """Luma standard deviation on the outside ring around the mask."""

    ring = context_ring_mask(mask_array, ring_px=ring_px)
    if not np.any(ring):
        return 0.0
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.std(gray[ring]))


def effective_low_freq_mae_threshold(
    base_threshold: float,
    ring_luma_std: float,
    *,
    textured_std: float = DEFAULT_TEXTURED_RING_LUMA_STD,
    scale_ref_std: float = DEFAULT_LOW_FREQ_STD_SCALE_REF,
) -> float | None:
    """Return the effective low-freq MAE threshold, or ``None`` to skip the gate.

    Flat/near-flat surroundings keep the base threshold. Highly textured rings
    (photo backgrounds) disable the gate so generative fills are not rejected for
    differing high-frequency content. Mid-range std scales the threshold up.
    """

    if base_threshold < 0:
        return None
    if ring_luma_std >= textured_std:
        return None
    if scale_ref_std <= 0:
        return base_threshold
    scale = max(1.0, ring_luma_std / scale_ref_std)
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
    "DEFAULT_LOW_FREQ_BLUR_SIGMA",
    "DEFAULT_LOW_FREQ_MAE_THRESHOLD",
    "DEFAULT_LOW_FREQ_STD_SCALE_REF",
    "DEFAULT_TEXTURED_RING_LUMA_STD",
    "GENERATIVE_COMPOSITE_EDGE_FEATHER_PX",
    "context_ring_luma_std",
    "context_ring_mask",
    "effective_low_freq_mae_threshold",
    "low_frequency_masked_mae",
    "seamless_clone_masked_edit",
]
