"""Post-generation OCR safety checks for generative inpainting."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import fitz
import numpy as np
from PIL import Image

from ..block_analysis import map_blocks_to_page_coordinates
from ..inpainting_masks import build_text_mask_image

logger = logging.getLogger(__name__)

ResidualChecker = Callable[[Image.Image, Image.Image], bool]


def residual_text_overlap_ratio(
    ocr_engine: Any,
    inpainted_image: Image.Image,
    mask_image: Image.Image,
    *,
    page_number: int = 1,
) -> float:
    """Fraction of the original mask still covered by OCR detections.

    Mirrors ``scripts/bench_inpaint.residual_text_detection_rate`` so production
    safety and bench metrics stay aligned.
    """

    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    original_mask_bool = mask_array > 0
    denominator = int(np.count_nonzero(original_mask_bool))
    if denominator == 0:
        return 0.0

    page_rect = fitz.Rect(0, 0, float(inpainted_image.width), float(inpainted_image.height))
    ocr_page_data = ocr_engine.extract_text_blocks(inpainted_image, page_number)
    if not ocr_page_data.blocks:
        return 0.0

    detected_blocks = map_blocks_to_page_coordinates(
        ocr_page_data.blocks,
        ocr_page_data.image.size,
        page_rect,
    )
    detected_mask_image = build_text_mask_image(
        detected_blocks,
        inpainted_image.size,
        page_rect,
        padding_px=0,
    )
    detected_mask_bool = np.array(detected_mask_image, dtype=np.uint8) > 0
    overlap = int(np.count_nonzero(detected_mask_bool & original_mask_bool))
    return overlap / denominator


def build_ocr_residual_checker(
    ocr_engine: Any,
    *,
    max_residual_ratio: float = 0.02,
) -> ResidualChecker:
    """Return a checker that rejects patches when OCR still finds text in the mask."""

    def _check(inpainted_image: Image.Image, mask_image: Image.Image) -> bool:
        try:
            ratio = residual_text_overlap_ratio(ocr_engine, inpainted_image, mask_image)
        except Exception as error:  # noqa: BLE001 - safety net must not crash conversion
            logger.warning("Generative OCR safety check failed open: %s", error)
            return False
        if ratio > max_residual_ratio:
            logger.info(
                "Generative OCR safety rejected patch (residual ratio %.4f > %.4f)",
                ratio,
                max_residual_ratio,
            )
            return False
        return True

    return _check


def assert_unmasked_pixels_unchanged(
    source: Image.Image,
    result: Image.Image,
    mask_image: Image.Image,
) -> None:
    """Hard invariant: pixels outside the mask must be byte-identical to source."""

    source_rgb = np.array(source.convert("RGB"), dtype=np.uint8)
    result_rgb = np.array(result.convert("RGB"), dtype=np.uint8)
    mask = np.array(mask_image.convert("L"), dtype=np.uint8) > 0
    if source_rgb.shape != result_rgb.shape:
        raise ValueError(
            f"Result size {result_rgb.shape} does not match source size {source_rgb.shape}."
        )
    if mask.shape[:2] != source_rgb.shape[:2]:
        raise ValueError("Mask dimensions do not match source image.")
    outside = ~mask
    if not np.array_equal(source_rgb[outside], result_rgb[outside]):
        raise AssertionError("Generative inpaint mutated pixels outside the mask.")


__all__ = [
    "ResidualChecker",
    "assert_unmasked_pixels_unchanged",
    "build_ocr_residual_checker",
    "residual_text_overlap_ratio",
]
