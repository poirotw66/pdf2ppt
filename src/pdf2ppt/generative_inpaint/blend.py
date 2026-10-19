"""Poisson / seamless blending and low-frequency quality gates for generative patches."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from ..inpainting_engines.compositing import _composite_lama_restoration_with_inference_mask
from ..inpainting_masks import estimate_low_texture_mask_fraction

logger = logging.getLogger(__name__)

GENERATIVE_COMPOSITE_EDGE_FEATHER_PX = 3
DEFAULT_LOW_FREQ_BLUR_SIGMA = 12.0
DEFAULT_LOW_FREQ_MAE_THRESHOLD = 4.0


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

    Uses ``cv2.seamlessClone`` (NORMAL_CLONE on low-texture masks, MIXED_CLONE
    otherwise). Falls back to distance-feather compositing when OpenCV rejects
    the clone (e.g. mask touching the crop border with no interior).
    """

    if source_rgb.shape != edited_rgb.shape:
        raise ValueError(
            f"source/edited shape mismatch: {source_rgb.shape} vs {edited_rgb.shape}"
        )
    binary = (mask_array > 0).astype(np.uint8)
    if not np.any(binary):
        return source_rgb.copy()

    if clone_mode is None:
        source_image = Image.fromarray(source_rgb, mode="RGB")
        mask_image = Image.fromarray(binary * 255, mode="L")
        low_texture = estimate_low_texture_mask_fraction(source_image, mask_image)
        clone_mode = cv2.NORMAL_CLONE if low_texture >= 0.75 else cv2.MIXED_CLONE

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
    "DEFAULT_LOW_FREQ_BLUR_SIGMA",
    "DEFAULT_LOW_FREQ_MAE_THRESHOLD",
    "GENERATIVE_COMPOSITE_EDGE_FEATHER_PX",
    "low_frequency_masked_mae",
    "seamless_clone_masked_edit",
]
