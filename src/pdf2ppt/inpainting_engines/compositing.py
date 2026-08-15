"""Alpha-blend compositing of LaMa-restored crops/pages back onto the original image.

Split out of the former monolithic ``inpainting_engines.py`` (Phase 1.3, pure move) -- no
blending math or defaults were changed while splitting the file.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import (
    DEFAULT_LAMA_COMPOSITE_ALPHA_GAIN,
    DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
    DEFAULT_LAMA_COMPOSITE_HARD_CORE_ERODE_PX,
    DEFAULT_LAMA_COMPOSITE_MASK_DILATE_PX,
    DEFAULT_LAMA_EDGE_FEATHER_PX,
    DEFAULT_LAMA_MASK_CLOSE_PX,
    DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD,
    DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD,
    DEFAULT_LAMA_PATCH_MAX_SIDE_PX,
)


def _finalize_lama_inpaint(
    source_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    working_mask: np.ndarray,
) -> np.ndarray:
    return _composite_lama_restoration(
        source_rgb,
        restored_rgb,
        working_mask,
        blend_sigma=DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
        alpha_gain=1.35,
        composite_dilate_px=1,
        hard_core_erode_px=0,
    )


def _expand_lama_inference_mask(mask_array: np.ndarray, *, dilate_px: int) -> np.ndarray:
    binary_mask = (mask_array > 0).astype(np.uint8)
    if dilate_px <= 0 or not np.any(binary_mask):
        return mask_array
    kernel_size = dilate_px * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(binary_mask, kernel, iterations=1) * 255


def _composite_lama_restoration(
    source_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    composite_mask: np.ndarray,
    *,
    inference_mask: np.ndarray | None = None,
    blend_sigma: float = DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
    alpha_gain: float = DEFAULT_LAMA_COMPOSITE_ALPHA_GAIN,
    composite_dilate_px: int = DEFAULT_LAMA_COMPOSITE_MASK_DILATE_PX,
    hard_core_erode_px: int = DEFAULT_LAMA_COMPOSITE_HARD_CORE_ERODE_PX,
) -> np.ndarray:
    if inference_mask is not None:
        return _composite_lama_restoration_with_inference_mask(
            source_rgb,
            restored_rgb,
            inference_mask,
            edge_feather_px=DEFAULT_LAMA_EDGE_FEATHER_PX,
        )
    alpha = _build_lama_composite_alpha(
        composite_mask,
        blend_sigma=blend_sigma,
        alpha_gain=alpha_gain,
        composite_dilate_px=composite_dilate_px,
        hard_core_erode_px=hard_core_erode_px,
    )
    if alpha is None:
        return source_rgb
    blended = source_rgb.astype(np.float32) * (1.0 - alpha) + restored_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _composite_lama_restoration_with_inference_mask(
    source_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    inference_mask: np.ndarray,
    *,
    edge_feather_px: int,
) -> np.ndarray:
    binary_mask = (inference_mask > 0).astype(np.uint8)
    if not np.any(binary_mask):
        return source_rgb

    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    feather = max(1, edge_feather_px)
    alpha = np.zeros_like(distance, dtype=np.float32)
    if max_distance <= feather:
        alpha[binary_mask > 0] = 1.0
    else:
        inner_threshold = max_distance - feather
        inside = binary_mask > 0
        alpha[inside & (distance >= inner_threshold)] = 1.0
        edge_band = inside & (distance < inner_threshold)
        alpha[edge_band] = np.clip(distance[edge_band] / feather, 0.0, 1.0)

    alpha = alpha[..., None]
    blended = source_rgb.astype(np.float32) * (1.0 - alpha) + restored_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _build_lama_composite_alpha(
    composite_mask: np.ndarray,
    *,
    blend_sigma: float,
    alpha_gain: float,
    composite_dilate_px: int,
    hard_core_erode_px: int,
) -> np.ndarray | None:
    binary_mask = (composite_mask > 0).astype(np.uint8)
    if not np.any(binary_mask):
        return None

    # Annotated explicitly: cv2.dilate's stub return type is wider (int/float dtype union)
    # than binary_mask's inferred uint8 dtype, and this local is conditionally reassigned
    # to that result below.
    expanded_mask: np.ndarray = binary_mask
    if composite_dilate_px > 0:
        kernel_size = composite_dilate_px * 2 + 1
        expanded_mask = cv2.dilate(
            binary_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )

    alpha = cv2.GaussianBlur(
        expanded_mask.astype(np.float32),
        (0, 0),
        sigmaX=blend_sigma,
        sigmaY=blend_sigma,
    )
    alpha = np.clip(alpha * alpha_gain, 0.0, 1.0)

    if hard_core_erode_px > 0:
        core_kernel_size = hard_core_erode_px * 2 + 1
        hard_core = cv2.erode(
            binary_mask,
            np.ones((core_kernel_size, core_kernel_size), dtype=np.uint8),
            iterations=1,
        )
        if np.any(hard_core):
            alpha[hard_core > 0] = 1.0

    return alpha[..., None]


def _lama_composite_debug_suffix(*, dilate_px: int) -> str:
    return (
        f" mask-close={DEFAULT_LAMA_MASK_CLOSE_PX}"
        f" inference-mask-dilate={dilate_px}"
        f" patch-threshold={DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD:.2f}"
        f" patch-hybrid-low-texture>={DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD:.2f}"
        f" patch-max-side={DEFAULT_LAMA_PATCH_MAX_SIDE_PX}"
    )




__all__ = [
    "_build_lama_composite_alpha",
    "_composite_lama_restoration",
    "_composite_lama_restoration_with_inference_mask",
    "_expand_lama_inference_mask",
    "_finalize_lama_inpaint",
    "_lama_composite_debug_suffix",
]
