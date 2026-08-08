# Backward-compatible re-export shim: `api.py` and `pipeline.py` import
# background/inpainting helpers from this module rather than the underlying
# `debug_artifacts` / `inpainting` modules directly. Nothing in this file is
# used locally -- every name below is intentionally re-exported.
from .debug_artifacts import build_mask_shapes, clip_to_image, write_debug_artifacts
from .inpainting import (
    BackgroundInpaintingEngine,
    BackgroundInpaintingError,
    BackgroundRenderResult,
    LamaOnnxCudaInpaintingEngine,
    LamaPytorchInpaintingEngine,
    OpenCvFastInpaintingEngine,
    WhiteBoxInpaintingEngine,
    build_text_mask_image,
    compute_mask_crop_box,
    estimate_background_complexity,
    mask_area_ratio,
    mask_text_regions_with_white_boxes,
    render_overlay_background,
    resolve_background_inpainting_engine,
)

__all__ = [
    "BackgroundInpaintingEngine",
    "BackgroundInpaintingError",
    "BackgroundRenderResult",
    "LamaOnnxCudaInpaintingEngine",
    "LamaPytorchInpaintingEngine",
    "OpenCvFastInpaintingEngine",
    "WhiteBoxInpaintingEngine",
    "build_mask_shapes",
    "build_text_mask_image",
    "clip_to_image",
    "compute_mask_crop_box",
    "estimate_background_complexity",
    "mask_area_ratio",
    "mask_text_regions_with_white_boxes",
    "render_overlay_background",
    "resolve_background_inpainting_engine",
    "write_debug_artifacts",
]
