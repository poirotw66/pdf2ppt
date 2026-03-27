from __future__ import annotations

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw

from .debug_artifacts import build_mask_shapes
from .models import TextBlock


def build_text_mask_image(
    text_blocks: list[TextBlock],
    image_size: tuple[int, int],
    page_rect: fitz.Rect,
    *,
    padding_px: int = 0,
) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in build_mask_shapes(text_blocks, image_size, page_rect):
        if shape["kind"] == "polygon":
            draw.polygon(shape["points"], fill=255)
        else:
            draw.rectangle(shape["bbox"], fill=255)

    if padding_px <= 0:
        return mask

    mask_array = np.array(mask, dtype=np.uint8)
    kernel_size = max(1, padding_px * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    expanded = cv2.dilate(mask_array, kernel, iterations=1)
    return Image.fromarray(expanded, mode="L")


def mask_text_regions_with_white_boxes(
    page_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
) -> Image.Image:
    from .inpainting_engines import WhiteBoxInpaintingEngine

    mask_image = build_text_mask_image(text_blocks, page_image.size, page_rect, padding_px=0)
    return WhiteBoxInpaintingEngine().inpaint(page_image, mask_image)


def mask_area_ratio(mask_array: np.ndarray) -> float:
    return float(np.count_nonzero(mask_array)) / float(mask_array.size)


def estimate_background_complexity(page_image: Image.Image, mask_image: Image.Image) -> float:
    image_array = np.array(page_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    kernel = np.ones((9, 9), dtype=np.uint8)
    outer_ring = cv2.dilate(mask_array, kernel, iterations=1)
    context_ring = np.logical_and(outer_ring > 0, mask_array == 0)
    if not np.any(context_ring):
        context_ring = mask_array == 0
    context_pixels = gray[context_ring]
    if context_pixels.size == 0:
        return 0.0

    luma_std = float(np.std(context_pixels))
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges[context_ring])) / float(context_pixels.size)
    variance_score = min(1.0, luma_std / 64.0)
    edge_score = min(1.0, edge_density / 0.12)
    return round(variance_score * 0.65 + edge_score * 0.35, 4)


def compute_mask_crop_box(mask_image: Image.Image, *, padding_px: int) -> tuple[int, int, int, int] | None:
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    points = cv2.findNonZero(mask_array)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    image_width, image_height = mask_image.size
    left = max(0, x - padding_px)
    top = max(0, y - padding_px)
    right = min(image_width, x + width + padding_px)
    bottom = min(image_height, y + height + padding_px)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


__all__ = [
    "build_text_mask_image",
    "compute_mask_crop_box",
    "estimate_background_complexity",
    "mask_area_ratio",
    "mask_text_regions_with_white_boxes",
]
