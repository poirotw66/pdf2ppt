"""The ``opencv-fast`` background inpainting engine.

Combines local low-texture/gradient surface fitting, an OpenCV Telea fallback for residual
regions, and structural line restoration. This module holds the engine's 20 tunable
hyperparameters together with the full prefill/residual/structural-line implementation.

Split out of the former monolithic ``inpainting_engines.py`` (Phase 1.3, pure move) -- no
defaults or algorithmic behavior were changed while splitting the file.
"""

from __future__ import annotations

import math
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image

from ..inpainting_masks import (
    DEFAULT_LOW_TEXTURE_CONTEXT_DILATE_PX,
    DEFAULT_LOW_TEXTURE_EDGE_THRESHOLD,
    DEFAULT_LOW_TEXTURE_STD_THRESHOLD,
)
from .base import (
    DEFAULT_RELAXED_QUADRATIC_COLOR_BIAS_MAX_DELTA,
    DEFAULT_RELAXED_QUADRATIC_EDGE_THRESHOLD,
    DEFAULT_RELAXED_QUADRATIC_MAX_HEIGHT_PX,
    DEFAULT_RELAXED_QUADRATIC_MIN_ASPECT_RATIO,
    DEFAULT_RELAXED_QUADRATIC_MIN_WIDTH_PX,
    DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_MAX_DELTA,
    DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_RESIDUAL_SCALE,
    DEFAULT_SMOOTH_GRADIENT_CONTEXT_GAP_PX,
    DEFAULT_SMOOTH_GRADIENT_EDGE_THRESHOLD,
    DEFAULT_SMOOTH_GRADIENT_QUADRATIC_RESIDUAL_THRESHOLD,
    DEFAULT_STRUCTURAL_LINE_MIN_KERNEL_PX,
    DEFAULT_STRUCTURAL_LINE_PADDING_PX,
    DEFAULT_TELEA_COMPACT_WIDE_EDGE_DENSITY_THRESHOLD,
    DEFAULT_TELEA_COMPACT_WIDE_MAX_HEIGHT_PX,
    DEFAULT_TELEA_COMPACT_WIDE_MAX_WIDTH_PX,
    DEFAULT_TELEA_COMPACT_WIDE_MIN_ASPECT_RATIO,
    DEFAULT_TELEA_DIRECTIONAL_SIDE_EDGE_MARGIN,
    DEFAULT_TELEA_DIRECTIONAL_SIDE_LUMA_MARGIN,
    DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_EDGE_MARGIN,
    DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_LUMA_MARGIN,
    DEFAULT_TELEA_GROUP_PROXIMITY_MAX_SCALE,
    DEFAULT_TELEA_GROUP_PROXIMITY_MIN_SCALE,
    DEFAULT_TELEA_GROUP_PROXIMITY_PX,
    DEFAULT_TELEA_ISOLATED_LABEL_EDGE_DENSITY_THRESHOLD,
    DEFAULT_TELEA_ISOLATED_LABEL_MAX_HEIGHT_PX,
    DEFAULT_TELEA_ISOLATED_LABEL_MAX_WIDTH_PX,
    DEFAULT_TELEA_ISOLATED_LABEL_MIN_ASPECT_RATIO,
    DEFAULT_TELEA_ISOLATED_LABEL_RADIUS_CAP,
    DEFAULT_TELEA_RADIUS_EDGE_DENSITY_MIN_FACTOR,
    DEFAULT_TELEA_RADIUS_EDGE_DENSITY_THRESHOLD,
    DEFAULT_TELEA_RADIUS_MAX_SCALE,
    DEFAULT_TELEA_RADIUS_MIN_SCALE,
    DEFAULT_TELEA_RADIUS_REFERENCE_SPAN_PX,
    DEFAULT_TELEA_SMALL_COMPONENT_MAX_SPAN_PX,
    BackgroundInpaintingEngine,
)


class OpenCvFastInpaintingEngine(BackgroundInpaintingEngine):
    name = "opencv-fast"

    def __init__(
        self,
        *,
        radius: float = 3.0,
        flat_background_std_threshold: float = DEFAULT_LOW_TEXTURE_STD_THRESHOLD,
        flat_background_edge_threshold: float = DEFAULT_LOW_TEXTURE_EDGE_THRESHOLD,
        context_dilate_px: int = DEFAULT_LOW_TEXTURE_CONTEXT_DILATE_PX,
        context_gap_px: int = DEFAULT_SMOOTH_GRADIENT_CONTEXT_GAP_PX,
        component_expand_px: int = 2,
        smooth_gradient_edge_threshold: float = DEFAULT_SMOOTH_GRADIENT_EDGE_THRESHOLD,
        smooth_gradient_residual_threshold: float = DEFAULT_SMOOTH_GRADIENT_QUADRATIC_RESIDUAL_THRESHOLD,
        smooth_gradient_color_bias_max_delta: float = DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_MAX_DELTA,
        smooth_gradient_color_bias_residual_scale: float = DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_RESIDUAL_SCALE,
        blend_sigma: float = 1.4,
        telea_radius_min_scale: float = DEFAULT_TELEA_RADIUS_MIN_SCALE,
        telea_radius_max_scale: float = DEFAULT_TELEA_RADIUS_MAX_SCALE,
        telea_radius_reference_span_px: float = DEFAULT_TELEA_RADIUS_REFERENCE_SPAN_PX,
        telea_radius_edge_density_threshold: float = DEFAULT_TELEA_RADIUS_EDGE_DENSITY_THRESHOLD,
        telea_radius_edge_density_min_factor: float = DEFAULT_TELEA_RADIUS_EDGE_DENSITY_MIN_FACTOR,
        telea_small_component_max_span_px: int = DEFAULT_TELEA_SMALL_COMPONENT_MAX_SPAN_PX,
        telea_group_proximity_px: int = DEFAULT_TELEA_GROUP_PROXIMITY_PX,
        telea_group_proximity_min_scale: float = DEFAULT_TELEA_GROUP_PROXIMITY_MIN_SCALE,
        telea_group_proximity_max_scale: float = DEFAULT_TELEA_GROUP_PROXIMITY_MAX_SCALE,
    ) -> None:
        self.radius = radius
        self.flat_background_std_threshold = flat_background_std_threshold
        self.flat_background_edge_threshold = flat_background_edge_threshold
        self.context_dilate_px = max(1, context_dilate_px)
        self.context_gap_px = max(0, context_gap_px)
        self.component_expand_px = max(0, component_expand_px)
        self.smooth_gradient_edge_threshold = max(0.0, smooth_gradient_edge_threshold)
        self.smooth_gradient_residual_threshold = max(0.0, smooth_gradient_residual_threshold)
        self.smooth_gradient_color_bias_max_delta = max(0.0, smooth_gradient_color_bias_max_delta)
        self.smooth_gradient_color_bias_residual_scale = max(1e-3, smooth_gradient_color_bias_residual_scale)
        self.blend_sigma = max(0.1, blend_sigma)
        self.telea_radius_min_scale = max(0.1, telea_radius_min_scale)
        self.telea_radius_max_scale = max(self.telea_radius_min_scale, telea_radius_max_scale)
        self.telea_radius_reference_span_px = max(1.0, telea_radius_reference_span_px)
        self.telea_radius_edge_density_threshold = max(1e-3, telea_radius_edge_density_threshold)
        self.telea_radius_edge_density_min_factor = float(np.clip(telea_radius_edge_density_min_factor, 0.1, 1.0))
        self.telea_small_component_max_span_px = max(1, telea_small_component_max_span_px)
        self.telea_group_proximity_px = max(0, telea_group_proximity_px)
        self.telea_group_proximity_min_scale = max(0.1, telea_group_proximity_min_scale)
        self.telea_group_proximity_max_scale = max(self.telea_group_proximity_min_scale, telea_group_proximity_max_scale)
        self._last_debug_note: str | None = None
        self._protected_line_mask: np.ndarray | None = None
        self._last_debug_images: dict[str, Image.Image] = {}

    @property
    def last_debug_note(self) -> str | None:
        return self._last_debug_note

    @property
    def last_debug_images(self) -> dict[str, Image.Image]:
        return self._last_debug_images

    def set_protected_line_mask(self, protected_line_mask: np.ndarray | None) -> None:
        if protected_line_mask is None:
            self._protected_line_mask = None
            return
        self._protected_line_mask = protected_line_mask.astype(bool, copy=True)

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        self._last_debug_note = None
        self._last_debug_images = {}
        mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
        if np.count_nonzero(mask_array) == 0:
            return page_image.convert("RGB").copy()
        source = cv2.cvtColor(np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        protected_prefill_ring_mask = np.zeros(mask_array.shape, dtype=np.uint8)
        prefill_result = _prefill_low_texture_regions(
            source,
            mask_array,
            protected_line_mask=self._protected_line_mask,
            protected_prefill_ring_mask=protected_prefill_ring_mask,
            flat_background_std_threshold=self.flat_background_std_threshold,
            flat_background_edge_threshold=self.flat_background_edge_threshold,
            context_dilate_px=self.context_dilate_px,
            context_gap_px=self.context_gap_px,
            component_expand_px=self.component_expand_px,
            smooth_gradient_edge_threshold=self.smooth_gradient_edge_threshold,
            smooth_gradient_residual_threshold=self.smooth_gradient_residual_threshold,
            smooth_gradient_color_bias_max_delta=self.smooth_gradient_color_bias_max_delta,
            smooth_gradient_color_bias_residual_scale=self.smooth_gradient_color_bias_residual_scale,
        )
        if len(prefill_result) == 2:
            prefilled_source, residual_mask = prefill_result
            prefilled_patches = []
        else:
            prefilled_source, residual_mask, prefilled_patches = prefill_result
        if np.count_nonzero(residual_mask) == 0:
            repaired = prefilled_source
            self._last_debug_note = "Residual Telea groups: 0 after surface prefill."
        else:
            repaired, diagnostics = _inpaint_residual_components(
                prefilled_source,
                residual_mask,
                protected_line_mask=self._protected_line_mask,
                base_radius=self.radius,
                min_radius=self.radius * self.telea_radius_min_scale,
                max_radius=self.radius * self.telea_radius_max_scale,
                reference_span_px=self.telea_radius_reference_span_px,
                edge_density_threshold=self.telea_radius_edge_density_threshold,
                edge_density_min_factor=self.telea_radius_edge_density_min_factor,
                small_component_max_span_px=self.telea_small_component_max_span_px,
                group_proximity_px=self.telea_group_proximity_px,
                group_proximity_min_scale=self.telea_group_proximity_min_scale,
                group_proximity_max_scale=self.telea_group_proximity_max_scale,
            )
            self._last_debug_note = _format_telea_group_diagnostics(diagnostics)
        repaired = _restore_low_texture_regions(
            source,
            repaired,
            mask_array,
            prefilled_patches=prefilled_patches,
            protected_line_mask=self._protected_line_mask,
            flat_background_std_threshold=self.flat_background_std_threshold,
            flat_background_edge_threshold=self.flat_background_edge_threshold,
            context_dilate_px=self.context_dilate_px,
            context_gap_px=self.context_gap_px,
            component_expand_px=self.component_expand_px,
            smooth_gradient_edge_threshold=self.smooth_gradient_edge_threshold,
            smooth_gradient_residual_threshold=self.smooth_gradient_residual_threshold,
            smooth_gradient_color_bias_max_delta=self.smooth_gradient_color_bias_max_delta,
            smooth_gradient_color_bias_residual_scale=self.smooth_gradient_color_bias_residual_scale,
            blend_sigma=self.blend_sigma,
        )
        repaired, restored_line_pixels = _restore_structural_line_regions(source, repaired, mask_array)
        if restored_line_pixels > 0:
            line_note = f"Structural line restore pixels: {restored_line_pixels}."
            self._last_debug_note = f"{self._last_debug_note} {line_note}" if self._last_debug_note else line_note
        if np.any(protected_prefill_ring_mask):
            self._last_debug_images["protected_prefill_ring_mask"] = Image.fromarray(protected_prefill_ring_mask, mode="L")
        return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))



def _prefill_low_texture_regions(
    source: np.ndarray,
    mask_array: np.ndarray,
    *,
    protected_line_mask: np.ndarray | None,
    protected_prefill_ring_mask: np.ndarray | None,
    flat_background_std_threshold: float,
    flat_background_edge_threshold: float,
    context_dilate_px: int,
    context_gap_px: int,
    component_expand_px: int,
    smooth_gradient_edge_threshold: float,
    smooth_gradient_residual_threshold: float,
    smooth_gradient_color_bias_max_delta: float,
    smooth_gradient_color_bias_residual_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int, np.ndarray, np.ndarray]]]:
    component_mask = (mask_array > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)  # type: ignore[call-overload]  # stub types arg 2 as `labels`, but the real binding dispatches a bare int here as `connectivity` (same stub gap as cv2.kmeans's bestLabels, db8002f)
    if component_count <= 1:
        return source, mask_array, []

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = np.ones((context_dilate_px * 2 + 1, context_dilate_px * 2 + 1), dtype=np.uint8)
    expand_kernel = None
    if component_expand_px > 0:
        expand_kernel = np.ones((component_expand_px * 2 + 1, component_expand_px * 2 + 1), dtype=np.uint8)
    gap_kernel = None
    if context_gap_px > 0:
        gap_kernel = np.ones((context_gap_px * 2 + 1, context_gap_px * 2 + 1), dtype=np.uint8)

    prefilled = source.astype(np.float32).copy()
    residual_mask = mask_array.copy()
    prefilled_patches: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for component_index, stat in enumerate(stats[1:], start=1):
        _, _, _, _, area = stat
        if area <= 0:
            continue

        component = (labels == component_index).astype(np.uint8)
        expanded_component, patch = _resolve_component_background_patch(
            source,
            gray,
            edges,
            component,
            protected_line_mask=protected_line_mask,
            protected_prefill_ring_mask=protected_prefill_ring_mask,
            kernel=kernel,
            expand_kernel=expand_kernel,
            gap_kernel=gap_kernel,
            flat_background_std_threshold=flat_background_std_threshold,
            flat_background_edge_threshold=flat_background_edge_threshold,
            context_dilate_px=context_dilate_px,
            smooth_gradient_edge_threshold=smooth_gradient_edge_threshold,
            smooth_gradient_residual_threshold=smooth_gradient_residual_threshold,
            smooth_gradient_color_bias_max_delta=smooth_gradient_color_bias_max_delta,
            smooth_gradient_color_bias_residual_scale=smooth_gradient_color_bias_residual_scale,
        )
        if patch is None:
            continue

        x0, y0, x1, y1, patch_values, _ = patch
        local_component = expanded_component[y0:y1, x0:x1].astype(bool)
        if not np.any(local_component):
            continue
        prefilled[y0:y1, x0:x1][local_component] = patch_values[local_component]
        residual_mask[expanded_component.astype(bool)] = 0
        prefilled_patches.append((x0, y0, x1, y1, local_component.copy(), patch_values.copy()))

    return np.clip(prefilled, 0, 255).astype(np.uint8), residual_mask, prefilled_patches


def _inpaint_residual_components(
    source: np.ndarray,
    residual_mask: np.ndarray,
    *,
    protected_line_mask: np.ndarray | None,
    base_radius: float,
    min_radius: float,
    max_radius: float,
    reference_span_px: float,
    edge_density_threshold: float,
    edge_density_min_factor: float,
    small_component_max_span_px: int,
    group_proximity_px: int,
    group_proximity_min_scale: float,
    group_proximity_max_scale: float,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    component_mask = (residual_mask > 0).astype(np.uint8)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(component_mask, 8)  # type: ignore[call-overload]  # stub types arg 2 as `labels`, but the real binding dispatches a bare int here as `connectivity` (same stub gap as cv2.kmeans's bestLabels, db8002f)
    if component_count <= 1:
        return source, []

    repaired = source.copy()
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    component_groups = _build_residual_component_groups(
        labels,
        component_count=component_count,
        small_component_max_span_px=small_component_max_span_px,
        group_proximity_px=group_proximity_px,
        group_proximity_min_scale=group_proximity_min_scale,
        group_proximity_max_scale=group_proximity_max_scale,
    )
    diagnostics: list[dict[str, float | int]] = []
    for group in component_groups:
        component = group["mask"]
        points = cv2.findNonZero(component)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        protected_nearby = False
        if protected_line_mask is not None and np.any(protected_line_mask):
            protected_padding = max(5, int(group["proximity_px"]) + 5)
            protected_kernel = np.ones((protected_padding * 2 + 1, protected_padding * 2 + 1), dtype=np.uint8)
            protected_vicinity = cv2.dilate(component, protected_kernel, iterations=1) > 0
            protected_nearby = bool(np.any(protected_line_mask & protected_vicinity))
        ring_mask = _build_component_ring_mask(component, padding_px=max(2, int(group["proximity_px"])))
        if protected_nearby and protected_line_mask is not None:
            ring_mask &= ~protected_line_mask
        ring_pixel_count = int(np.count_nonzero(ring_mask))
        edge_density = 0.0
        if ring_pixel_count > 0:
            edge_density = float(np.count_nonzero(edges[ring_mask])) / float(ring_pixel_count)
        component_radius = _resolve_component_telea_radius(
            width=width,
            height=height,
            base_radius=base_radius,
            min_radius=min_radius,
            max_radius=max_radius,
            reference_span_px=reference_span_px,
            edge_density=edge_density,
            edge_density_threshold=edge_density_threshold,
            edge_density_min_factor=edge_density_min_factor,
        )
        component_radius = _clamp_isolated_label_telea_radius(
            component_radius,
            width=width,
            height=height,
            edge_density=edge_density,
            group_size=int(group["group_size"]),
            proximity_px=int(group["proximity_px"]),
        )
        if protected_nearby:
            component_radius = max(min_radius, component_radius * 0.65)
            padding_scale = 1.1
        else:
            padding_scale = 2.0
        padding = max(2, int(math.ceil(component_radius * padding_scale)))
        # source.shape is typed as the variable-length tuple[int, ...]; the crop-bounds
        # helpers below take a fixed 2-tuple, so build one explicitly instead of slicing.
        image_shape = (source.shape[0], source.shape[1])
        if protected_nearby and protected_line_mask is not None:
            x0, y0, x1, y1 = _resolve_protected_inpaint_crop_bounds(
                protected_line_mask,
                component_bbox=(x, y, width, height),
                image_shape=image_shape,
                base_padding=padding,
            )
        elif _is_compact_wide_residual_component(width=width, height=height, edge_density=edge_density):
            x0, y0, x1, y1 = _resolve_directional_inpaint_crop_bounds(
                gray,
                edges,
                component_bbox=(x, y, width, height),
                image_shape=image_shape,
                base_padding=padding,
            )
        else:
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(source.shape[1], x + width + padding)
            y1 = min(source.shape[0], y + height + padding)

        local_source = repaired[y0:y1, x0:x1]
        local_mask = (component[y0:y1, x0:x1] * 255).astype(np.uint8)
        local_repaired = cv2.inpaint(local_source, local_mask, component_radius, cv2.INPAINT_TELEA)
        local_component = local_mask > 0
        local_target = repaired[y0:y1, x0:x1]
        local_target[local_component] = local_repaired[local_component]
        diagnostics.append(
            {
                "group_size": int(group["group_size"]),
                "proximity_px": int(group["proximity_px"]),
                "edge_density": float(edge_density),
                "final_radius": float(component_radius),
                "protected_nearby": int(protected_nearby),
            }
        )

    return repaired, diagnostics


class _SmallResidualComponent(TypedDict):
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    span: int


class _ResidualComponentGroup(TypedDict):
    mask: np.ndarray
    group_size: int
    proximity_px: int


def _build_residual_component_groups(
    labels: np.ndarray,
    *,
    component_count: int,
    small_component_max_span_px: int,
    group_proximity_px: int,
    group_proximity_min_scale: float,
    group_proximity_max_scale: float,
) -> list[_ResidualComponentGroup]:
    small_components: list[_SmallResidualComponent] = []
    groups: list[_ResidualComponentGroup] = []
    for component_index in range(1, component_count):
        component = (labels == component_index).astype(np.uint8)
        points = cv2.findNonZero(component)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        span = max(width, height)
        if span <= small_component_max_span_px:
            small_components.append({
                "mask": component,
                "bbox": (x, y, width, height),
                "span": span,
            })
        else:
            groups.append({"mask": component, "group_size": 1, "proximity_px": 0})

    if not small_components:
        return groups

    parent = list(range(len(small_components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(small_components):
        left_bbox = left["bbox"]
        left_span = int(left["span"])
        for right_index in range(left_index + 1, len(small_components)):
            right = small_components[right_index]
            right_bbox = right["bbox"]
            right_span = int(right["span"])
            adaptive_proximity = _resolve_group_proximity_px(
                base_proximity_px=group_proximity_px,
                span=max(left_span, right_span),
                reference_span_px=small_component_max_span_px,
                min_scale=group_proximity_min_scale,
                max_scale=group_proximity_max_scale,
            )
            if _bbox_gap_px(left_bbox, right_bbox) <= adaptive_proximity:
                union(left_index, right_index)

    grouped_components: dict[int, list[_SmallResidualComponent]] = {}
    for index, component in enumerate(small_components):
        grouped_components.setdefault(find(index), []).append(component)

    for members in grouped_components.values():
        group_mask = np.zeros_like(labels, dtype=np.uint8)
        max_span = 0
        for member in members:
            group_mask = np.maximum(group_mask, member["mask"])
            max_span = max(max_span, int(member["span"]))
        groups.append(
            {
                "mask": group_mask,
                "group_size": len(members),
                "proximity_px": _resolve_group_proximity_px(
                    base_proximity_px=group_proximity_px,
                    span=max_span,
                    reference_span_px=small_component_max_span_px,
                    min_scale=group_proximity_min_scale,
                    max_scale=group_proximity_max_scale,
                ),
            }
        )

    return groups


def _resolve_group_proximity_px(
    *,
    base_proximity_px: int,
    span: int,
    reference_span_px: int,
    min_scale: float,
    max_scale: float,
) -> int:
    if base_proximity_px <= 0:
        return 0
    scale = float(np.clip(span / max(1.0, float(reference_span_px)), min_scale, max_scale))
    return max(1, int(round(base_proximity_px * scale)))


def _bbox_gap_px(
    left_bbox: tuple[int, int, int, int] | np.ndarray,
    right_bbox: tuple[int, int, int, int] | np.ndarray,
) -> int:
    left_x, left_y, left_width, left_height = left_bbox
    right_x, right_y, right_width, right_height = right_bbox
    left_right = left_x + left_width
    left_bottom = left_y + left_height
    right_right = right_x + right_width
    right_bottom = right_y + right_height
    x_gap = max(0, right_x - left_right, left_x - right_right)
    y_gap = max(0, right_y - left_bottom, left_y - right_bottom)
    return max(x_gap, y_gap)


def _build_component_ring_mask(component: np.ndarray, *, padding_px: int) -> np.ndarray:
    kernel = np.ones((padding_px * 2 + 1, padding_px * 2 + 1), dtype=np.uint8)
    outer = cv2.dilate(component, kernel, iterations=1).astype(bool)
    inner = component.astype(bool)
    return outer & (~inner)


def _resolve_component_telea_radius(
    *,
    width: int,
    height: int,
    base_radius: float,
    min_radius: float,
    max_radius: float,
    reference_span_px: float,
    edge_density: float,
    edge_density_threshold: float,
    edge_density_min_factor: float,
) -> float:
    component_span = _resolve_component_telea_effective_span(
        width=width,
        height=height,
        edge_density=edge_density,
    )
    size_scale = component_span / reference_span_px
    density_ratio = min(1.0, max(0.0, edge_density) / edge_density_threshold)
    edge_scale = 1.0 - density_ratio * (1.0 - edge_density_min_factor)
    return float(np.clip(base_radius * size_scale * edge_scale, min_radius, max_radius))


def _resolve_component_telea_effective_span(*, width: int, height: int, edge_density: float) -> float:
    span = float(max(width, height))
    if height <= 0:
        return span
    if _is_compact_wide_residual_component(width=width, height=height, edge_density=edge_density):
        return float(math.sqrt(float(width) * float(height)))
    return span


def _is_compact_wide_residual_component(*, width: int, height: int, edge_density: float) -> bool:
    if height <= 0:
        return False
    aspect_ratio = float(width) / float(height)
    return (
        width <= DEFAULT_TELEA_COMPACT_WIDE_MAX_WIDTH_PX
        and height <= DEFAULT_TELEA_COMPACT_WIDE_MAX_HEIGHT_PX
        and aspect_ratio >= DEFAULT_TELEA_COMPACT_WIDE_MIN_ASPECT_RATIO
        and edge_density <= DEFAULT_TELEA_COMPACT_WIDE_EDGE_DENSITY_THRESHOLD
    )


def _clamp_isolated_label_telea_radius(
    radius: float,
    *,
    width: int,
    height: int,
    edge_density: float,
    group_size: int,
    proximity_px: int,
) -> float:
    if height <= 0:
        return radius
    aspect_ratio = float(width) / float(height)
    if group_size != 1 or proximity_px != 0:
        return radius
    if width > DEFAULT_TELEA_ISOLATED_LABEL_MAX_WIDTH_PX:
        return radius
    if height > DEFAULT_TELEA_ISOLATED_LABEL_MAX_HEIGHT_PX:
        return radius
    if aspect_ratio < DEFAULT_TELEA_ISOLATED_LABEL_MIN_ASPECT_RATIO:
        return radius
    if edge_density > DEFAULT_TELEA_ISOLATED_LABEL_EDGE_DENSITY_THRESHOLD:
        return radius
    return min(radius, DEFAULT_TELEA_ISOLATED_LABEL_RADIUS_CAP)


def _resolve_directional_inpaint_crop_bounds(
    gray: np.ndarray,
    edges: np.ndarray,
    *,
    component_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    base_padding: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = component_bbox
    image_height, image_width = image_shape
    x0 = max(0, x - base_padding)
    y0 = max(0, y - base_padding)
    x1 = min(image_width, x + width + base_padding)
    y1 = min(image_height, y + height + base_padding)

    left_region = gray[y:y + height, x0:x]
    right_region = gray[y:y + height, x + width:x1]
    top_region = gray[y0:y, x:x + width]
    bottom_region = gray[y + height:y1, x:x + width]
    left_edges = edges[y:y + height, x0:x]
    right_edges = edges[y:y + height, x + width:x1]
    top_edges = edges[y0:y, x:x + width]
    bottom_edges = edges[y + height:y1, x:x + width]

    vertical_luma = _region_percentile_mean(top_region, bottom_region, percentile=10.0)
    vertical_edge = _region_edge_density_mean(top_edges, bottom_edges)
    ultra_near_padding = 0
    very_near_padding = max(1, int(math.ceil(base_padding * 0.18)))
    near_padding = max(2, int(math.ceil(base_padding * 0.45)))
    far_padding = max(base_padding, int(math.ceil(base_padding * 1.15)))

    left_padding = _resolve_directional_side_padding(
        left_region,
        left_edges,
        vertical_luma=vertical_luma,
        vertical_edge=vertical_edge,
        ultra_near_padding=ultra_near_padding,
        very_near_padding=very_near_padding,
        near_padding=near_padding,
        far_padding=far_padding,
    )
    right_padding = _resolve_directional_side_padding(
        right_region,
        right_edges,
        vertical_luma=vertical_luma,
        vertical_edge=vertical_edge,
        ultra_near_padding=ultra_near_padding,
        very_near_padding=very_near_padding,
        near_padding=near_padding,
        far_padding=far_padding,
    )
    top_padding = far_padding
    bottom_padding = far_padding

    return (
        max(0, x - left_padding),
        max(0, y - top_padding),
        min(image_width, x + width + right_padding),
        min(image_height, y + height + bottom_padding),
    )


def _resolve_directional_side_padding(
    region: np.ndarray,
    edge_region: np.ndarray,
    *,
    vertical_luma: float,
    vertical_edge: float,
    ultra_near_padding: int,
    very_near_padding: int,
    near_padding: int,
    far_padding: int,
) -> int:
    if region.size == 0:
        return very_near_padding
    side_luma = float(np.percentile(region, 10))
    side_edge = float(np.count_nonzero(edge_region)) / float(max(1, region.size))
    if side_luma + DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_LUMA_MARGIN < vertical_luma:
        return ultra_near_padding
    if side_edge > vertical_edge + DEFAULT_TELEA_DIRECTIONAL_SIDE_STRONG_EDGE_MARGIN:
        return ultra_near_padding
    if side_luma + DEFAULT_TELEA_DIRECTIONAL_SIDE_LUMA_MARGIN < vertical_luma:
        return near_padding
    if side_edge > vertical_edge + DEFAULT_TELEA_DIRECTIONAL_SIDE_EDGE_MARGIN:
        return near_padding
    return far_padding


def _region_percentile_mean(first: np.ndarray, second: np.ndarray, *, percentile: float) -> float:
    values: list[float] = []
    if first.size > 0:
        values.append(float(np.percentile(first, percentile)))
    if second.size > 0:
        values.append(float(np.percentile(second, percentile)))
    if not values:
        return 255.0
    return float(np.mean(values))


def _region_edge_density_mean(first: np.ndarray, second: np.ndarray) -> float:
    values: list[float] = []
    for region in (first, second):
        if region.size > 0:
            values.append(float(np.count_nonzero(region)) / float(region.size))
    if not values:
        return 0.0
    return float(np.mean(values))


def _resolve_protected_inpaint_crop_bounds(
    protected_line_mask: np.ndarray,
    *,
    component_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    base_padding: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = component_bbox
    image_height, image_width = image_shape
    score_padding = max(base_padding, 8)
    symmetric_x0 = max(0, x - score_padding)
    symmetric_y0 = max(0, y - score_padding)
    symmetric_x1 = min(image_width, x + width + score_padding)
    symmetric_y1 = min(image_height, y + height + score_padding)
    near_padding = max(2, int(math.ceil(base_padding * 0.35)))
    far_padding = max(near_padding + 1, base_padding)

    left_score = int(np.count_nonzero(protected_line_mask[symmetric_y0:symmetric_y1, symmetric_x0:x]))
    right_score = int(np.count_nonzero(protected_line_mask[symmetric_y0:symmetric_y1, x + width : symmetric_x1]))
    top_score = int(np.count_nonzero(protected_line_mask[symmetric_y0:y, symmetric_x0:symmetric_x1]))
    bottom_score = int(np.count_nonzero(protected_line_mask[y + height : symmetric_y1, symmetric_x0:symmetric_x1]))

    left_padding = near_padding if left_score > 0 else far_padding
    right_padding = near_padding if right_score > 0 else far_padding
    top_padding = near_padding if top_score > 0 else far_padding
    bottom_padding = near_padding if bottom_score > 0 else far_padding

    x0 = max(0, x - left_padding)
    y0 = max(0, y - top_padding)
    x1 = min(image_width, x + width + right_padding)
    y1 = min(image_height, y + height + bottom_padding)
    return x0, y0, x1, y1


def _format_telea_group_diagnostics(diagnostics: list[dict[str, float | int]]) -> str:
    if not diagnostics:
        return "Residual Telea groups: 0."
    edge_values = [float(item["edge_density"]) for item in diagnostics]
    group_sizes = [int(item["group_size"]) for item in diagnostics]
    radius_values = [float(item["final_radius"]) for item in diagnostics]
    proximity_values = [int(item["proximity_px"]) for item in diagnostics]
    protected_count = sum(int(item.get("protected_nearby", 0)) for item in diagnostics)
    return (
        f"Residual Telea groups: {len(diagnostics)}; "
        f"group size {min(group_sizes)}-{max(group_sizes)}; "
        f"adaptive proximity {min(proximity_values)}-{max(proximity_values)} px; "
        f"edge density {min(edge_values):.3f}-{max(edge_values):.3f}; "
        f"final radius {min(radius_values):.2f}-{max(radius_values):.2f}; "
        f"protected groups {protected_count}."
    )


def _restore_structural_line_regions(
    source: np.ndarray,
    repaired: np.ndarray,
    mask_array: np.ndarray,
    *,
    padding_px: int = DEFAULT_STRUCTURAL_LINE_PADDING_PX,
    min_kernel_px: int = DEFAULT_STRUCTURAL_LINE_MIN_KERNEL_PX,
) -> tuple[np.ndarray, int]:
    if np.count_nonzero(mask_array) == 0:
        return repaired, 0

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    binary[mask_array > 0] = 0
    horizontal_kernel_len = max(min_kernel_px, source.shape[1] // 30)
    vertical_kernel_len = max(min_kernel_px, source.shape[0] // 20)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))
    horizontal_lines = _filter_structural_line_candidates(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel),
        orientation="horizontal",
        min_span_px=horizontal_kernel_len,
    )
    vertical_lines = _filter_structural_line_candidates(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel),
        orientation="vertical",
        min_span_px=vertical_kernel_len,
    )
    line_mask = cv2.bitwise_or(horizontal_lines, vertical_lines) > 0
    if not np.any(line_mask):
        return repaired, 0

    grid_region_mask = _build_grid_region_mask(horizontal_lines, vertical_lines, mask_array)
    if not np.any(grid_region_mask):
        return repaired, 0

    result = repaired.copy()
    restored_pixels = 0
    restored_pixels += _bridge_horizontal_line_gaps(
        source,
        result,
        horizontal_lines,
        mask_array,
        grid_region_mask,
        search_radius=max(horizontal_kernel_len, padding_px * 4),
    )
    restored_pixels += _bridge_vertical_line_gaps(
        source,
        result,
        vertical_lines,
        mask_array,
        grid_region_mask,
        search_radius=max(vertical_kernel_len, padding_px * 4),
    )
    if restored_pixels == 0:
        return repaired, 0
    return result, restored_pixels


def _bridge_horizontal_line_gaps(
    source: np.ndarray,
    result: np.ndarray,
    horizontal_lines: np.ndarray,
    mask_array: np.ndarray,
    grid_region_mask: np.ndarray,
    *,
    search_radius: int,
) -> int:
    restored_pixels = 0
    height, width = mask_array.shape
    for row in range(height):
        if not np.any(horizontal_lines[row] > 0) or not np.any(mask_array[row] > 0):
            continue
        for start, end in _iter_true_runs(mask_array[row] > 0):
            if not np.any(grid_region_mask[row, start : end + 1]):
                continue
            left_slice = slice(max(0, start - search_radius), start)
            right_slice = slice(end + 1, min(width, end + 1 + search_radius))
            left_support = np.flatnonzero(horizontal_lines[row, left_slice] > 0)
            right_support = np.flatnonzero(horizontal_lines[row, right_slice] > 0)
            if left_support.size == 0 or right_support.size == 0:
                continue
            left_positions = left_support + left_slice.start
            right_positions = right_support + right_slice.start
            support_values = np.concatenate(
                [source[row, left_positions].astype(np.float32), source[row, right_positions].astype(np.float32)],
                axis=0,
            )
            fill_color = np.mean(support_values, axis=0)
            result[row, start : end + 1][mask_array[row, start : end + 1] > 0] = fill_color
            restored_pixels += end - start + 1
    return restored_pixels


def _bridge_vertical_line_gaps(
    source: np.ndarray,
    result: np.ndarray,
    vertical_lines: np.ndarray,
    mask_array: np.ndarray,
    grid_region_mask: np.ndarray,
    *,
    search_radius: int,
) -> int:
    restored_pixels = 0
    height, width = mask_array.shape
    for col in range(width):
        if not np.any(vertical_lines[:, col] > 0) or not np.any(mask_array[:, col] > 0):
            continue
        for start, end in _iter_true_runs(mask_array[:, col] > 0):
            if not np.any(grid_region_mask[start : end + 1, col]):
                continue
            top_slice = slice(max(0, start - search_radius), start)
            bottom_slice = slice(end + 1, min(height, end + 1 + search_radius))
            top_support = np.flatnonzero(vertical_lines[top_slice, col] > 0)
            bottom_support = np.flatnonzero(vertical_lines[bottom_slice, col] > 0)
            if top_support.size == 0 or bottom_support.size == 0:
                continue
            top_positions = top_support + top_slice.start
            bottom_positions = bottom_support + bottom_slice.start
            support_values = np.concatenate(
                [source[top_positions, col].astype(np.float32), source[bottom_positions, col].astype(np.float32)],
                axis=0,
            )
            fill_color = np.mean(support_values, axis=0)
            result[start : end + 1, col][mask_array[start : end + 1, col] > 0] = fill_color
            restored_pixels += end - start + 1
    return restored_pixels


def _iter_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _build_grid_region_mask(
    horizontal_lines: np.ndarray,
    vertical_lines: np.ndarray,
    mask_array: np.ndarray,
) -> np.ndarray:
    horizontal_bool = horizontal_lines > 0
    vertical_bool = vertical_lines > 0
    if not np.any(horizontal_bool) or not np.any(vertical_bool):
        return np.zeros_like(mask_array, dtype=bool)

    intersection_seed = horizontal_bool & vertical_bool
    if not np.any(intersection_seed):
        horizontal_kernel = np.ones((1, max(3, mask_array.shape[1] // 80)), dtype=np.uint8)
        vertical_kernel = np.ones((max(3, mask_array.shape[0] // 80), 1), dtype=np.uint8)
        expanded_horizontal = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1) > 0
        expanded_vertical = cv2.dilate(vertical_lines, vertical_kernel, iterations=1) > 0
        intersection_seed = expanded_horizontal & expanded_vertical
        if not np.any(intersection_seed):
            return np.zeros_like(mask_array, dtype=bool)

    region_kernel = np.ones((max(5, mask_array.shape[0] // 40), max(5, mask_array.shape[1] // 40)), dtype=np.uint8)
    grid_region = cv2.dilate(intersection_seed.astype(np.uint8), region_kernel, iterations=1) > 0
    line_region = cv2.dilate((horizontal_bool | vertical_bool).astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
    mask_vicinity = cv2.dilate((mask_array > 0).astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1) > 0
    return grid_region & line_region & mask_vicinity


def _filter_structural_line_candidates(
    line_mask: np.ndarray,
    *,
    orientation: str,
    min_span_px: int,
) -> np.ndarray:
    component_mask = (line_mask > 0).astype(np.uint8)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(component_mask, 8)  # type: ignore[call-overload]  # stub types arg 2 as `labels`, but the real binding dispatches a bare int here as `connectivity` (same stub gap as cv2.kmeans's bestLabels, db8002f)
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


def _restore_low_texture_regions(
    source: np.ndarray,
    repaired: np.ndarray,
    mask_array: np.ndarray,
    *,
    prefilled_patches: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] | None,
    protected_line_mask: np.ndarray | None,
    flat_background_std_threshold: float,
    flat_background_edge_threshold: float,
    context_dilate_px: int,
    context_gap_px: int,
    component_expand_px: int,
    smooth_gradient_edge_threshold: float,
    smooth_gradient_residual_threshold: float,
    smooth_gradient_color_bias_max_delta: float,
    smooth_gradient_color_bias_residual_scale: float,
    blend_sigma: float,
) -> np.ndarray:
    if prefilled_patches:
        blended = repaired.astype(np.float32)
        for x0, y0, x1, y1, local_component, patch_values in prefilled_patches:
            alpha = cv2.GaussianBlur(local_component.astype(np.float32), (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma)
            alpha = np.clip(alpha * 1.5, 0.0, 1.0)[..., None]
            blended[y0:y1, x0:x1] = blended[y0:y1, x0:x1] * (1.0 - alpha) + patch_values * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    component_mask = (mask_array > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)  # type: ignore[call-overload]  # stub types arg 2 as `labels`, but the real binding dispatches a bare int here as `connectivity` (same stub gap as cv2.kmeans's bestLabels, db8002f)
    if component_count <= 1:
        return repaired

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    kernel = np.ones((context_dilate_px * 2 + 1, context_dilate_px * 2 + 1), dtype=np.uint8)
    expand_kernel = None
    if component_expand_px > 0:
        expand_kernel = np.ones((component_expand_px * 2 + 1, component_expand_px * 2 + 1), dtype=np.uint8)
    gap_kernel = None
    if context_gap_px > 0:
        gap_kernel = np.ones((context_gap_px * 2 + 1, context_gap_px * 2 + 1), dtype=np.uint8)
    blended = repaired.astype(np.float32)
    protected_prefill_ring_mask = None
    if protected_line_mask is not None:
        protected_prefill_ring_mask = np.zeros(mask_array.shape, dtype=np.uint8)

    for component_index, stat in enumerate(stats[1:], start=1):
        _, _, _, _, area = stat
        if area <= 0:
            continue

        component = (labels == component_index).astype(np.uint8)
        expanded_component, patch = _resolve_component_background_patch(
            source,
            gray,
            edges,
            component,
            protected_line_mask=protected_line_mask,
            protected_prefill_ring_mask=protected_prefill_ring_mask,
            kernel=kernel,
            expand_kernel=expand_kernel,
            gap_kernel=gap_kernel,
            flat_background_std_threshold=flat_background_std_threshold,
            flat_background_edge_threshold=flat_background_edge_threshold,
            context_dilate_px=context_dilate_px,
            smooth_gradient_edge_threshold=smooth_gradient_edge_threshold,
            smooth_gradient_residual_threshold=smooth_gradient_residual_threshold,
            smooth_gradient_color_bias_max_delta=smooth_gradient_color_bias_max_delta,
            smooth_gradient_color_bias_residual_scale=smooth_gradient_color_bias_residual_scale,
        )
        if patch is None:
            continue

        x0, y0, x1, y1, patch_values, _ = patch
        local_component = expanded_component[y0:y1, x0:x1].astype(np.float32)
        alpha = cv2.GaussianBlur(local_component, (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma)
        alpha = np.clip(alpha * 1.5, 0.0, 1.0)[..., None]
        blended[y0:y1, x0:x1] = (
            blended[y0:y1, x0:x1] * (1.0 - alpha) + patch_values * alpha
        )

    return np.clip(blended, 0, 255).astype(np.uint8)


def _resolve_component_background_patch(
    source: np.ndarray,
    gray: np.ndarray,
    edges: np.ndarray,
    component: np.ndarray,
    *,
    protected_line_mask: np.ndarray | None,
    protected_prefill_ring_mask: np.ndarray | None,
    kernel: np.ndarray,
    expand_kernel: np.ndarray | None,
    gap_kernel: np.ndarray | None,
    flat_background_std_threshold: float,
    flat_background_edge_threshold: float,
    context_dilate_px: int,
    smooth_gradient_edge_threshold: float,
    smooth_gradient_residual_threshold: float,
    smooth_gradient_color_bias_max_delta: float,
    smooth_gradient_color_bias_residual_scale: float,
) -> tuple[np.ndarray, tuple[int, int, int, int, np.ndarray, float] | None]:
    expanded_component = component
    if expand_kernel is not None:
        expanded_component = cv2.dilate(component, expand_kernel, iterations=1)

    context_component = expanded_component
    if gap_kernel is not None:
        context_component = cv2.dilate(expanded_component, gap_kernel, iterations=1)

    ring_mask = cv2.dilate(context_component, kernel, iterations=1).astype(bool) & (~context_component.astype(bool))
    if protected_line_mask is not None and np.any(protected_line_mask):
        protected_vicinity = cv2.dilate(component, np.ones((11, 11), dtype=np.uint8), iterations=1) > 0
        if np.any(protected_line_mask & protected_vicinity):
            ring_mask &= ~protected_line_mask
            if protected_prefill_ring_mask is not None:
                protected_prefill_ring_mask[ring_mask] = 255
    ring_pixel_count = int(np.count_nonzero(ring_mask))
    if ring_pixel_count < 64:
        return expanded_component, None

    context_values = gray[ring_mask]
    luma_std = float(np.std(context_values))
    edge_density = float(np.count_nonzero(edges[ring_mask])) / float(ring_pixel_count)
    component_points = cv2.findNonZero(expanded_component)
    if component_points is None:
        return expanded_component, None
    _, _, component_width, component_height = cv2.boundingRect(component_points)
    patch = None
    patch_model: str | None = None
    if luma_std <= flat_background_std_threshold and edge_density <= flat_background_edge_threshold:
        patch = _fit_component_background_surface(
            source,
            expanded_component,
            ring_mask,
            context_dilate_px=context_dilate_px,
            model="plane",
        )
        patch_model = "plane"
    elif edge_density <= smooth_gradient_edge_threshold:
        patch = _fit_component_background_surface(
            source,
            expanded_component,
            ring_mask,
            context_dilate_px=context_dilate_px,
            model="quadratic",
        )
        if patch is not None and patch[-1] > smooth_gradient_residual_threshold:
            patch = None
        else:
            patch_model = "quadratic"
    elif _should_try_relaxed_quadratic_prefill(
        width=component_width,
        height=component_height,
        edge_density=edge_density,
    ):
        patch = _fit_component_background_surface(
            source,
            expanded_component,
            ring_mask,
            context_dilate_px=context_dilate_px,
            model="quadratic",
        )
        if patch is not None and patch[-1] > smooth_gradient_residual_threshold:
            patch = None
        else:
            patch_model = "quadratic-relaxed"

    if patch is None:
        return expanded_component, None

    x0, y0, x1, y1, patch_values, patch_residual = patch
    color_bias_max_delta = smooth_gradient_color_bias_max_delta
    if patch_model == "quadratic-relaxed" and color_bias_max_delta <= 0.0:
        color_bias_max_delta = DEFAULT_RELAXED_QUADRATIC_COLOR_BIAS_MAX_DELTA

    if patch_model in {"quadratic", "quadratic-relaxed"} and color_bias_max_delta > 0.0:
        local_ring = ring_mask[y0:y1, x0:x1]
        source_patch = source[y0:y1, x0:x1]
        local_component_mask = expanded_component[y0:y1, x0:x1].astype(bool)
        effective_max_delta = color_bias_max_delta * min(
            1.0,
            patch_residual / smooth_gradient_color_bias_residual_scale,
        )
        patch_values = _align_patch_mean_to_ring_background(
            patch_values,
            source_patch,
            local_component_mask,
            local_ring,
            max_delta=effective_max_delta,
        )
        patch = (x0, y0, x1, y1, patch_values, patch_residual)
    return expanded_component, patch


def _should_try_relaxed_quadratic_prefill(*, width: int, height: int, edge_density: float) -> bool:
    if width < DEFAULT_RELAXED_QUADRATIC_MIN_WIDTH_PX:
        return False
    if height > DEFAULT_RELAXED_QUADRATIC_MAX_HEIGHT_PX:
        return False
    if width / max(1.0, float(height)) < DEFAULT_RELAXED_QUADRATIC_MIN_ASPECT_RATIO:
        return False
    return edge_density <= DEFAULT_RELAXED_QUADRATIC_EDGE_THRESHOLD


def _fit_component_background_surface(
    source: np.ndarray,
    component: np.ndarray,
    ring_mask: np.ndarray,
    *,
    context_dilate_px: int,
    model: str,
) -> tuple[int, int, int, int, np.ndarray, float] | None:
    points = cv2.findNonZero(component)
    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(points)
    image_height, image_width = source.shape[:2]
    padding = context_dilate_px * 2
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image_width, x + width + padding)
    y1 = min(image_height, y + height + padding)

    local_ring = ring_mask[y0:y1, x0:x1]
    ring_y, ring_x = np.nonzero(local_ring)
    if ring_x.size < 32:
        return None

    design_matrix = _build_surface_design_matrix(ring_x.astype(np.float32), ring_y.astype(np.float32), model=model)
    patch_height = y1 - y0
    patch_width = x1 - x0
    grid_y, grid_x = np.indices((patch_height, patch_width), dtype=np.float32)
    patch_design_matrix = _build_surface_design_matrix(grid_x.reshape(-1), grid_y.reshape(-1), model=model)
    patch_values = np.empty((patch_height, patch_width, source.shape[2]), dtype=np.float32)
    source_patch = source[y0:y1, x0:x1]
    residuals: list[float] = []
    for channel_index in range(source.shape[2]):
        channel_values = source_patch[:, :, channel_index][local_ring].astype(np.float32)
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, channel_values, rcond=None)
        fitted_ring = design_matrix @ coefficients
        residuals.append(float(np.mean(np.abs(fitted_ring - channel_values))))
        patch_values[:, :, channel_index] = (patch_design_matrix @ coefficients).reshape(patch_height, patch_width)

    return x0, y0, x1, y1, np.clip(patch_values, 0.0, 255.0), float(np.mean(residuals))


def _build_surface_design_matrix(x: np.ndarray, y: np.ndarray, *, model: str) -> np.ndarray:
    if model == "plane":
        return np.stack(
            [
                x,
                y,
                np.ones_like(x, dtype=np.float32),
            ],
            axis=1,
        )

    if model == "quadratic":
        return np.stack(
            [
                x * x,
                y * y,
                x * y,
                x,
                y,
                np.ones_like(x, dtype=np.float32),
            ],
            axis=1,
        )

    raise ValueError(f"Unsupported surface model: {model}")


def _align_patch_mean_to_ring_background(
    patch_values: np.ndarray,
    source_patch: np.ndarray,
    local_component_mask: np.ndarray,
    local_ring: np.ndarray,
    *,
    max_delta: float,
) -> np.ndarray:
    if max_delta <= 0.0 or not np.any(local_ring) or not np.any(local_component_mask):
        return patch_values

    corrected = patch_values.copy()
    for channel_index in range(patch_values.shape[2]):
        ring_values = source_patch[:, :, channel_index][local_ring].astype(np.float32)
        component_values = patch_values[:, :, channel_index][local_component_mask].astype(np.float32)
        if ring_values.size == 0 or component_values.size == 0:
            continue
        ring_values.sort()
        trim = ring_values.size // 10
        if trim > 0 and ring_values.size > trim * 2:
            ring_values = ring_values[trim:-trim]
        ring_mean = float(np.mean(ring_values))
        component_mean = float(np.mean(component_values))
        delta = ring_mean - component_mean
        if abs(delta) < 1e-3:
            continue
        corrected[:, :, channel_index] = np.clip(
            corrected[:, :, channel_index] + np.clip(delta, -max_delta, max_delta),
            0.0,
            255.0,
        )

    return corrected



__all__ = [
    "OpenCvFastInpaintingEngine",
    "_align_patch_mean_to_ring_background",
    "_bbox_gap_px",
    "_bridge_horizontal_line_gaps",
    "_bridge_vertical_line_gaps",
    "_build_component_ring_mask",
    "_build_grid_region_mask",
    "_build_residual_component_groups",
    "_build_surface_design_matrix",
    "_clamp_isolated_label_telea_radius",
    "_filter_structural_line_candidates",
    "_fit_component_background_surface",
    "_format_telea_group_diagnostics",
    "_inpaint_residual_components",
    "_is_compact_wide_residual_component",
    "_iter_true_runs",
    "_prefill_low_texture_regions",
    "_region_edge_density_mean",
    "_region_percentile_mean",
    "_resolve_component_background_patch",
    "_resolve_component_telea_effective_span",
    "_resolve_component_telea_radius",
    "_resolve_directional_inpaint_crop_bounds",
    "_resolve_directional_side_padding",
    "_resolve_group_proximity_px",
    "_resolve_protected_inpaint_crop_bounds",
    "_restore_low_texture_regions",
    "_restore_structural_line_regions",
    "_should_try_relaxed_quadratic_prefill",
]
