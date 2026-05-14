from __future__ import annotations

from dataclasses import dataclass

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw

from .debug_artifacts import build_mask_shapes
from .models import TextBlock


DEFAULT_LOW_TEXTURE_STD_THRESHOLD = 4.0
DEFAULT_LOW_TEXTURE_EDGE_THRESHOLD = 0.01
DEFAULT_LOW_TEXTURE_CONTEXT_DILATE_PX = 8
DEFAULT_LOW_TEXTURE_MIN_CONTEXT_PIXELS = 64
DEFAULT_TABLE_LINE_MIN_KERNEL_PX = 12
DEFAULT_TABLE_LINE_SHRINK_KERNEL_PX = 3
DEFAULT_TABLE_LINE_BOUNDARY_BAND_PX = 5
DEFAULT_TABLE_LINE_MAX_TEXT_OVERLAP_RATIO = 0.85
DEFAULT_TABLE_LINE_MIN_OUTSIDE_PIXELS = 12
DEFAULT_TABLE_LINE_MIN_OUTSIDE_RATIO = 0.08


@dataclass(slots=True)
class InpaintingMaskRefinementResult:
    mask_image: Image.Image
    raw_mask_image: Image.Image
    refined_mask_image: Image.Image
    table_line_mask_image: Image.Image | None
    grid_line_mask_image: Image.Image | None


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


def refine_text_mask_for_inpainting(
    page_image: Image.Image,
    text_blocks: list[TextBlock],
    image_size: tuple[int, int],
    page_rect: fitz.Rect,
    *,
    padding_px: int = 0,
) -> Image.Image:
    return build_refined_text_mask_for_inpainting(
        page_image,
        text_blocks,
        image_size,
        page_rect,
        padding_px=padding_px,
    ).mask_image


def build_refined_text_mask_for_inpainting(
    page_image: Image.Image,
    text_blocks: list[TextBlock],
    image_size: tuple[int, int],
    page_rect: fitz.Rect,
    *,
    padding_px: int = 0,
) -> InpaintingMaskRefinementResult:
    base_mask_image = build_text_mask_image(text_blocks, image_size, page_rect, padding_px=0)
    base_mask_array = np.array(base_mask_image, dtype=np.uint8)
    if np.count_nonzero(base_mask_array) == 0:
        return InpaintingMaskRefinementResult(
            mask_image=base_mask_image,
            raw_mask_image=base_mask_image,
            refined_mask_image=base_mask_image,
            table_line_mask_image=None,
            grid_line_mask_image=None,
        )

    raw_mask_image = build_text_mask_image(text_blocks, image_size, page_rect, padding_px=padding_px)
    refined_mask_image = raw_mask_image
    refined_mask_array = np.array(refined_mask_image, dtype=np.uint8)
    ocr_blocks = [block for block in text_blocks if block.source == "ocr"]
    if not ocr_blocks:
        return InpaintingMaskRefinementResult(
            mask_image=refined_mask_image,
            raw_mask_image=raw_mask_image,
            refined_mask_image=refined_mask_image,
            table_line_mask_image=None,
            grid_line_mask_image=None,
        )

    ocr_base_mask_image = build_text_mask_image(ocr_blocks, image_size, page_rect, padding_px=0)
    ocr_base_mask = np.array(ocr_base_mask_image, dtype=np.uint8) > 0
    horizontal_line_mask, vertical_line_mask = _detect_table_line_orientation_masks(page_image)
    horizontal_line_mask = _filter_text_overlapping_line_components(horizontal_line_mask, ocr_base_mask)
    vertical_line_mask = _filter_text_overlapping_line_components(vertical_line_mask, ocr_base_mask)
    table_line_mask = horizontal_line_mask | vertical_line_mask
    table_line_mask_image = Image.fromarray((table_line_mask.astype(np.uint8) * 255), mode="L")
    grid_line_mask = _build_grid_line_mask(horizontal_line_mask, vertical_line_mask)
    grid_line_mask_image = Image.fromarray((grid_line_mask.astype(np.uint8) * 255), mode="L")
    if not np.any(table_line_mask):
        return InpaintingMaskRefinementResult(
            mask_image=refined_mask_image,
            raw_mask_image=raw_mask_image,
            refined_mask_image=refined_mask_image,
            table_line_mask_image=table_line_mask_image,
            grid_line_mask_image=grid_line_mask_image,
        )

    ocr_vicinity_image = build_text_mask_image(
        ocr_blocks,
        image_size,
        page_rect,
        padding_px=max(1, padding_px + 1),
    )
    ocr_vicinity = np.array(ocr_vicinity_image, dtype=np.uint8) > 0
    padded_fringe = (refined_mask_array > 0) & ~(base_mask_array > 0)
    ocr_boundary_band = _build_mask_boundary_band(ocr_base_mask, band_px=DEFAULT_TABLE_LINE_BOUNDARY_BAND_PX)

    line_vicinity = cv2.dilate(table_line_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    shrink_region = padded_fringe & ocr_vicinity & line_vicinity
    if np.any(shrink_region):
        shrunken_mask = cv2.erode(
            refined_mask_array,
            np.ones((DEFAULT_TABLE_LINE_SHRINK_KERNEL_PX, DEFAULT_TABLE_LINE_SHRINK_KERNEL_PX), dtype=np.uint8),
            iterations=1,
        )
        refined_mask_array[shrink_region] = shrunken_mask[shrink_region]

    refined_mask_array[padded_fringe & table_line_mask] = 0
    refined_mask_array[ocr_boundary_band & line_vicinity] = 0
    refined_mask_array[base_mask_array > 0] = 255
    refined_mask_array[ocr_boundary_band & line_vicinity] = 0
    refined_mask_image = Image.fromarray(refined_mask_array, mode="L")
    return InpaintingMaskRefinementResult(
        mask_image=refined_mask_image,
        raw_mask_image=raw_mask_image,
        refined_mask_image=refined_mask_image,
        table_line_mask_image=table_line_mask_image,
        grid_line_mask_image=grid_line_mask_image,
    )


def _build_mask_boundary_band(mask: np.ndarray, *, band_px: int) -> np.ndarray:
    if band_px <= 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    kernel_size = band_px * 2 + 1
    eroded = cv2.erode(mask.astype(np.uint8) * 255, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    return mask & ~(eroded > 0)


def _detect_table_line_orientation_masks(page_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(np.array(page_image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    height, width = gray.shape
    horizontal_kernel_len = max(DEFAULT_TABLE_LINE_MIN_KERNEL_PX, width // 24)
    vertical_kernel_len = max(DEFAULT_TABLE_LINE_MIN_KERNEL_PX, height // 24)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, horizontal_kernel_len), dtype=np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((vertical_kernel_len, 1), dtype=np.uint8))
    horizontal = _filter_line_components(horizontal, orientation="horizontal", min_span_px=horizontal_kernel_len)
    vertical = _filter_line_components(vertical, orientation="vertical", min_span_px=vertical_kernel_len)
    return horizontal > 0, vertical > 0


def _build_grid_line_mask(horizontal_line_mask: np.ndarray, vertical_line_mask: np.ndarray) -> np.ndarray:
    if not np.any(horizontal_line_mask) or not np.any(vertical_line_mask):
        return np.zeros_like(horizontal_line_mask, dtype=bool)

    expanded_horizontal = cv2.dilate(horizontal_line_mask.astype(np.uint8), np.ones((3, 9), dtype=np.uint8), iterations=1) > 0
    expanded_vertical = cv2.dilate(vertical_line_mask.astype(np.uint8), np.ones((9, 3), dtype=np.uint8), iterations=1) > 0
    intersection_seed = expanded_horizontal & expanded_vertical
    if not np.any(intersection_seed):
        return np.zeros_like(horizontal_line_mask, dtype=bool)

    grid_region = cv2.dilate(intersection_seed.astype(np.uint8), np.ones((21, 21), dtype=np.uint8), iterations=1) > 0
    return (horizontal_line_mask | vertical_line_mask) & grid_region


def _filter_line_components(line_mask: np.ndarray, *, orientation: str, min_span_px: int) -> np.ndarray:
    component_mask = (line_mask > 0).astype(np.uint8)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    if component_count <= 1:
        return line_mask

    filtered = np.zeros_like(line_mask)
    for component_index in range(1, component_count):
        component = (labels == component_index).astype(np.uint8)
        points = cv2.findNonZero(component)
        if points is None:
            continue
        _, _, width, height = cv2.boundingRect(points)
        if orientation == "horizontal":
            if width < min_span_px or width < height * 4:
                continue
        else:
            if height < min_span_px or height < width * 4:
                continue
        filtered[component.astype(bool)] = 255
    return filtered


def _filter_text_overlapping_line_components(
    line_mask: np.ndarray,
    text_mask: np.ndarray,
    *,
    max_text_overlap_ratio: float = DEFAULT_TABLE_LINE_MAX_TEXT_OVERLAP_RATIO,
    min_outside_pixels: int = DEFAULT_TABLE_LINE_MIN_OUTSIDE_PIXELS,
    min_outside_ratio: float = DEFAULT_TABLE_LINE_MIN_OUTSIDE_RATIO,
) -> np.ndarray:
    if not np.any(line_mask) or not np.any(text_mask):
        return line_mask

    component_mask = (line_mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    if component_count <= 1:
        return line_mask

    filtered = np.zeros_like(line_mask)
    text_mask = text_mask.astype(bool)
    for component_index, stat in enumerate(stats[1:], start=1):
        area = int(stat[cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == component_index
        overlap = int(np.count_nonzero(component & text_mask))
        outside = area - overlap
        outside_floor = max(min_outside_pixels, int(np.ceil(area * min_outside_ratio)))
        if overlap / float(area) > max_text_overlap_ratio and outside < outside_floor:
            continue
        filtered[component] = 255
    return filtered


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


def estimate_low_texture_mask_fraction(
    page_image: Image.Image,
    mask_image: Image.Image,
    *,
    std_threshold: float = DEFAULT_LOW_TEXTURE_STD_THRESHOLD,
    edge_threshold: float = DEFAULT_LOW_TEXTURE_EDGE_THRESHOLD,
    context_dilate_px: int = DEFAULT_LOW_TEXTURE_CONTEXT_DILATE_PX,
    min_context_pixels: int = DEFAULT_LOW_TEXTURE_MIN_CONTEXT_PIXELS,
) -> float:
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    component_mask = (mask_array > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    if component_count <= 1:
        return 0.0

    image_array = np.array(page_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = np.ones((context_dilate_px * 2 + 1, context_dilate_px * 2 + 1), dtype=np.uint8)

    low_texture_area = 0
    valid_area = 0
    for component_index, stat in enumerate(stats[1:], start=1):
        area = int(stat[cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component = (labels == component_index).astype(np.uint8)
        ring_mask = cv2.dilate(component, kernel, iterations=1).astype(bool) & (~component.astype(bool))
        ring_pixel_count = int(np.count_nonzero(ring_mask))
        if ring_pixel_count < min_context_pixels:
            continue

        valid_area += area
        luma_std = float(np.std(gray[ring_mask]))
        edge_density = float(np.count_nonzero(edges[ring_mask])) / float(ring_pixel_count)
        if luma_std <= std_threshold and edge_density <= edge_threshold:
            low_texture_area += area

    if valid_area <= 0:
        return 0.0
    return round(float(low_texture_area) / float(valid_area), 4)


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
    "build_refined_text_mask_for_inpainting",
    "compute_mask_crop_box",
    "DEFAULT_LOW_TEXTURE_CONTEXT_DILATE_PX",
    "DEFAULT_LOW_TEXTURE_EDGE_THRESHOLD",
    "DEFAULT_LOW_TEXTURE_MIN_CONTEXT_PIXELS",
    "DEFAULT_LOW_TEXTURE_STD_THRESHOLD",
    "InpaintingMaskRefinementResult",
    "estimate_background_complexity",
    "estimate_low_texture_mask_fraction",
    "mask_area_ratio",
    "mask_text_regions_with_white_boxes",
    "refine_text_mask_for_inpainting",
]
