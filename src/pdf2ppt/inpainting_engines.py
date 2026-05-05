from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .inpainting_masks import (
    DEFAULT_LOW_TEXTURE_CONTEXT_DILATE_PX,
    DEFAULT_LOW_TEXTURE_EDGE_THRESHOLD,
    DEFAULT_LOW_TEXTURE_STD_THRESHOLD,
)

logger = logging.getLogger(__name__)


DEFAULT_SMOOTH_GRADIENT_CONTEXT_GAP_PX = 6
DEFAULT_SMOOTH_GRADIENT_EDGE_THRESHOLD = 0.015
DEFAULT_SMOOTH_GRADIENT_QUADRATIC_RESIDUAL_THRESHOLD = 16.0
DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_MAX_DELTA = 0.0
DEFAULT_SMOOTH_GRADIENT_COLOR_BIAS_RESIDUAL_SCALE = 4.0


@dataclass(slots=True)
class BackgroundRenderResult:
    image: Image.Image
    engine_name: str | None
    note: str | None
    mask_image: Image.Image | None = None
    debug_images: dict[str, Image.Image] = field(default_factory=dict, repr=False)


class BackgroundInpaintingError(RuntimeError):
    pass


class BackgroundInpaintingEngine:
    name = "base"

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        raise NotImplementedError


class WhiteBoxInpaintingEngine(BackgroundInpaintingEngine):
    name = "white-box"

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        mask_array = np.array(mask_image.convert("L")) > 0
        result = np.array(page_image.convert("RGB"), copy=True)
        result[mask_array] = 255
        return Image.fromarray(result, mode="RGB")


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

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
        if np.count_nonzero(mask_array) == 0:
            return page_image.convert("RGB").copy()
        source = cv2.cvtColor(np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        repaired = cv2.inpaint(source, mask_array, self.radius, cv2.INPAINT_TELEA)
        repaired = _restore_low_texture_regions(
            source,
            repaired,
            mask_array,
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
        return Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))


def _restore_low_texture_regions(
    source: np.ndarray,
    repaired: np.ndarray,
    mask_array: np.ndarray,
    *,
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
    component_mask = (mask_array > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, 8)
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

    for component_index, stat in enumerate(stats[1:], start=1):
        _, _, _, _, area = stat
        if area <= 0:
            continue

        component = (labels == component_index).astype(np.uint8)
        expanded_component = component
        if expand_kernel is not None:
            expanded_component = cv2.dilate(component, expand_kernel, iterations=1)

        context_component = expanded_component
        if gap_kernel is not None:
            context_component = cv2.dilate(expanded_component, gap_kernel, iterations=1)

        ring_mask = cv2.dilate(context_component, kernel, iterations=1).astype(bool) & (~context_component.astype(bool))
        ring_pixel_count = int(np.count_nonzero(ring_mask))
        if ring_pixel_count < 64:
            continue

        context_values = gray[ring_mask]
        luma_std = float(np.std(context_values))
        edge_density = float(np.count_nonzero(edges[ring_mask])) / float(ring_pixel_count)
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

        if patch is None:
            continue

        x0, y0, x1, y1, patch_values, patch_residual = patch
        if patch_model == "quadratic" and smooth_gradient_color_bias_max_delta > 0.0:
            local_ring = ring_mask[y0:y1, x0:x1]
            source_patch = source[y0:y1, x0:x1]
            local_component_mask = expanded_component[y0:y1, x0:x1].astype(bool)
            effective_max_delta = smooth_gradient_color_bias_max_delta * min(
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
        local_component = expanded_component[y0:y1, x0:x1].astype(np.float32)
        alpha = cv2.GaussianBlur(local_component, (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma)
        alpha = np.clip(alpha * 1.5, 0.0, 1.0)[..., None]
        blended[y0:y1, x0:x1] = (
            blended[y0:y1, x0:x1] * (1.0 - alpha) + patch_values * alpha
        )

    return np.clip(blended, 0, 255).astype(np.uint8)


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
    "BackgroundInpaintingEngine",
    "BackgroundInpaintingError",
    "BackgroundRenderResult",
    "OpenCvFastInpaintingEngine",
    "WhiteBoxInpaintingEngine",
]
