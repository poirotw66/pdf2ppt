"""Patch-group splitting and per-patch hybrid (opencv-fast vs. LaMa) routing.

Handles carving a large or multi-component mask into per-region "patch groups" for LaMa
inference, deciding per patch whether opencv-fast is good enough (the "hybrid" fast path), and
compositing patch results back onto the full page.

Split out of the former monolithic ``inpainting_engines.py`` (Phase 1.3, pure move) -- no
thresholds or patch-selection logic were changed while splitting the file.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..inpainting_masks import estimate_background_complexity, estimate_low_texture_mask_fraction, mask_area_ratio
from .base import (
    DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
    DEFAULT_LAMA_MASK_CLOSE_PX,
    DEFAULT_LAMA_PATCH_CONTEXT_PADDING_PX,
    DEFAULT_LAMA_PATCH_HYBRID_COMPLEXITY_THRESHOLD,
    DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD,
    DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD,
    DEFAULT_LAMA_PATCH_MAX_COMPONENTS_FOR_FULL_PAGE,
    DEFAULT_LAMA_PATCH_MERGE_GAP_PX,
    BackgroundInpaintingError,
)
from .compositing import _composite_lama_restoration
from .opencv_fast import OpenCvFastInpaintingEngine

logger = logging.getLogger(__name__)


def _prepare_lama_working_mask(mask_array: np.ndarray, *, close_px: int = DEFAULT_LAMA_MASK_CLOSE_PX) -> np.ndarray:
    # Annotated explicitly: cv2.morphologyEx's stub return type is wider (int/float dtype
    # union) than the inferred uint8 dtype, and this local is conditionally reassigned to
    # that result below.
    binary_mask: np.ndarray = (mask_array > 0).astype(np.uint8)
    if not np.any(binary_mask):
        return mask_array
    if close_px > 0:
        kernel_size = close_px * 2 + 1
        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
    return binary_mask * 255


def _should_use_lama_patch_inpaint(mask_array: np.ndarray) -> bool:
    if mask_area_ratio(mask_array) > DEFAULT_LAMA_PATCH_INPAINT_MASK_RATIO_THRESHOLD:
        return True
    component_count, _, _, _ = cv2.connectedComponentsWithStats((mask_array > 0).astype(np.uint8), 8)  # type: ignore[call-overload]  # stub types arg 2 as `labels`, but the real binding dispatches a bare int here as `connectivity` (same stub gap as cv2.kmeans's bestLabels, db8002f)
    return component_count > DEFAULT_LAMA_PATCH_MAX_COMPONENTS_FOR_FULL_PAGE + 1


def _boxes_overlap_with_gap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    *,
    gap_px: int,
) -> bool:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    return not (
        left_x1 + gap_px < right_x0
        or right_x1 + gap_px < left_x0
        or left_y1 + gap_px < right_y0
        or right_y1 + gap_px < left_y0
    )


def _merge_lama_patch_boxes(
    boxes: list[tuple[int, int, int, int, int]],
    *,
    merge_gap_px: int,
) -> list[tuple[int, int, int, int, list[int]]]:
    if not boxes:
        return []

    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left_box in enumerate(boxes):
        for right_index in range(left_index + 1, len(boxes)):
            if _boxes_overlap_with_gap(left_box[:4], boxes[right_index][:4], gap_px=merge_gap_px):
                union(left_index, right_index)

    grouped: dict[int, list[int]] = {}
    for index, box in enumerate(boxes):
        grouped.setdefault(find(index), []).append(box[4])

    merged_boxes: list[tuple[int, int, int, int, list[int]]] = []
    for member_indices in grouped.values():
        member_boxes = [box for box in boxes if box[4] in member_indices]
        x0 = min(box[0] for box in member_boxes)
        y0 = min(box[1] for box in member_boxes)
        x1 = max(box[2] for box in member_boxes)
        y1 = max(box[3] for box in member_boxes)
        merged_boxes.append((x0, y0, x1, y1, member_indices))
    return merged_boxes


def _build_lama_patch_groups(
    working_mask: np.ndarray,
    *,
    context_padding_px: int,
    merge_gap_px: int,
) -> list[tuple[int, int, int, int, np.ndarray]]:
    height, width = working_mask.shape[:2]
    binary_mask = (working_mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)  # type: ignore[call-overload]  # stub types arg 2 as `labels`, but the real binding dispatches a bare int here as `connectivity` (same stub gap as cv2.kmeans's bestLabels, db8002f)
    if component_count <= 1:
        return []

    initial_boxes: list[tuple[int, int, int, int, int]] = []
    for component_index in range(1, component_count):
        x, y, box_width, box_height, _ = stats[component_index]
        if box_width <= 0 or box_height <= 0:
            continue
        initial_boxes.append(
            (
                max(0, int(x) - context_padding_px),
                max(0, int(y) - context_padding_px),
                min(width, int(x + box_width) + context_padding_px),
                min(height, int(y + box_height) + context_padding_px),
                component_index,
            )
        )

    merged_boxes = _merge_lama_patch_boxes(initial_boxes, merge_gap_px=merge_gap_px)
    patch_groups: list[tuple[int, int, int, int, np.ndarray]] = []
    for x0, y0, x1, y1, member_indices in merged_boxes:
        group_mask = np.zeros((height, width), dtype=np.uint8)
        for component_index in member_indices:
            group_mask[labels == component_index] = 255
        patch_groups.append((x0, y0, x1, y1, group_mask))
    return patch_groups


def _composite_lama_patch_into_page(
    page_rgb: np.ndarray,
    patch_rgb: np.ndarray,
    crop_mask: np.ndarray,
    *,
    origin: tuple[int, int],
) -> None:
    x0, y0 = origin
    patch_height, patch_width = patch_rgb.shape[:2]
    region = page_rgb[y0 : y0 + patch_height, x0 : x0 + patch_width]
    blended = _composite_lama_restoration(
        region,
        patch_rgb,
        crop_mask,
        blend_sigma=DEFAULT_LAMA_COMPOSITE_BLEND_SIGMA,
        alpha_gain=1.35,
        composite_dilate_px=1,
        hard_core_erode_px=0,
    )
    page_rgb[y0 : y0 + patch_height, x0 : x0 + patch_width] = blended


def _should_use_opencv_for_lama_patch(
    crop_source: np.ndarray,
    crop_mask: np.ndarray,
) -> bool:
    crop_page = Image.fromarray(crop_source, mode="RGB")
    crop_mask_image = Image.fromarray(crop_mask, mode="L")
    low_texture_fraction = estimate_low_texture_mask_fraction(crop_page, crop_mask_image)
    if low_texture_fraction >= DEFAULT_LAMA_PATCH_HYBRID_LOW_TEXTURE_THRESHOLD:
        return True
    complexity = estimate_background_complexity(crop_page, crop_mask_image)
    return (
        complexity <= DEFAULT_LAMA_PATCH_HYBRID_COMPLEXITY_THRESHOLD
        and mask_area_ratio(crop_mask) <= 0.25
    )


def _inpaint_opencv_fast_crop(source_rgb: np.ndarray, working_mask: np.ndarray) -> np.ndarray:
    crop_page = Image.fromarray(source_rgb, mode="RGB")
    crop_mask_image = Image.fromarray(working_mask, mode="L")
    return np.array(OpenCvFastInpaintingEngine().inpaint(crop_page, crop_mask_image), dtype=np.uint8)


def _inpaint_lama_crop_with_hybrid(
    crop_source: np.ndarray,
    crop_working_mask: np.ndarray,
    run_lama_crop_inpaint: Any,
    *,
    use_patch_hybrid: bool,
) -> tuple[np.ndarray, str]:
    if use_patch_hybrid and _should_use_opencv_for_lama_patch(crop_source, crop_working_mask):
        return _inpaint_opencv_fast_crop(crop_source, crop_working_mask), "opencv-fast"
    return run_lama_crop_inpaint(crop_source, crop_working_mask), "lama"


def _inpaint_lama_page_by_patches(
    source_rgb: np.ndarray,
    working_mask: np.ndarray,
    run_crop_inpaint: Any,
    *,
    use_patch_hybrid: bool,
    run_crops_batch_inpaint: Any | None = None,
) -> tuple[np.ndarray, str]:
    patch_groups = _build_lama_patch_groups(
        working_mask,
        context_padding_px=DEFAULT_LAMA_PATCH_CONTEXT_PADDING_PX,
        merge_gap_px=DEFAULT_LAMA_PATCH_MERGE_GAP_PX,
    )
    if not patch_groups:
        return source_rgb.copy(), "patch-mode groups=0"

    result = source_rgb.copy()
    opencv_patch_count = 0
    lama_patch_count = 0
    lama_jobs: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []

    for x0, y0, x1, y1, group_mask in patch_groups:
        crop_source = source_rgb[y0:y1, x0:x1]
        crop_mask = group_mask[y0:y1, x0:x1]
        if not np.any(crop_mask):
            continue
        crop_working_mask = _prepare_lama_working_mask(crop_mask)
        if use_patch_hybrid and _should_use_opencv_for_lama_patch(crop_source, crop_working_mask):
            crop_restored = _inpaint_opencv_fast_crop(crop_source, crop_working_mask)
            opencv_patch_count += 1
            _composite_lama_patch_into_page(
                result,
                crop_restored,
                crop_working_mask,
                origin=(x0, y0),
            )
            continue
        lama_jobs.append((x0, y0, x1, y1, crop_source, crop_working_mask))

    if lama_jobs:
        use_batch = run_crops_batch_inpaint is not None and len(lama_jobs) > 1
        if use_batch:
            logger.info(
                "LaMa patch mode: batch inpainting %s/%s patch groups (hybrid=%s)",
                len(lama_jobs),
                len(patch_groups),
                use_patch_hybrid,
            )
            batch_crops = [(job[4], job[5]) for job in lama_jobs]
            # use_batch is only True when `run_crops_batch_inpaint is not None` (see its
            # definition above); make that invariant explicit for the type checker.
            assert run_crops_batch_inpaint is not None
            batch_restored = run_crops_batch_inpaint(batch_crops)
            if len(batch_restored) != len(lama_jobs):
                raise BackgroundInpaintingError(
                    f"LaMa batch inpaint returned {len(batch_restored)} crops for {len(lama_jobs)} jobs."
                )
            for (x0, y0, _x1, _y1, _crop_source, crop_working_mask), crop_restored in zip(
                lama_jobs, batch_restored, strict=True
            ):
                lama_patch_count += 1
                _composite_lama_patch_into_page(
                    result,
                    crop_restored,
                    crop_working_mask,
                    origin=(x0, y0),
                )
        else:
            for x0, y0, _x1, _y1, crop_source, crop_working_mask in lama_jobs:
                crop_restored = run_crop_inpaint(crop_source, crop_working_mask)
                lama_patch_count += 1
                _composite_lama_patch_into_page(
                    result,
                    crop_restored,
                    crop_working_mask,
                    origin=(x0, y0),
                )

    hybrid_note = (
        f" hybrid-opencv={opencv_patch_count} hybrid-lama={lama_patch_count}"
        if use_patch_hybrid
        else ""
    )
    batch_note = " batch-lama" if lama_jobs and run_crops_batch_inpaint is not None and len(lama_jobs) > 1 else ""
    patch_note = (
        f"patch-mode groups={len(patch_groups)}{hybrid_note}{batch_note} "
        f"mask-ratio={mask_area_ratio(working_mask):.4f}"
    )
    return result, patch_note




__all__ = [
    "_boxes_overlap_with_gap",
    "_build_lama_patch_groups",
    "_composite_lama_patch_into_page",
    "_inpaint_lama_crop_with_hybrid",
    "_inpaint_lama_page_by_patches",
    "_inpaint_opencv_fast_crop",
    "_merge_lama_patch_boxes",
    "_prepare_lama_working_mask",
    "_should_use_lama_patch_inpaint",
    "_should_use_opencv_for_lama_patch",
]
