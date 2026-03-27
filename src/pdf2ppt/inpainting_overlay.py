from __future__ import annotations

import logging
import shutil

import fitz
import numpy as np
from PIL import Image

from .core import ConversionOptions
from .inpainting_engines import (
    BackgroundInpaintingEngine,
    BackgroundInpaintingError,
    BackgroundRenderResult,
    DiffusionLocalInpaintingEngine,
    OpenCvFastInpaintingEngine,
    WhiteBoxInpaintingEngine,
)
from .inpainting_masks import (
    build_text_mask_image,
    estimate_background_complexity,
    mask_area_ratio,
)
from .models import TextBlock

logger = logging.getLogger(__name__)


def render_overlay_background(
    page_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
    *,
    options: ConversionOptions,
) -> BackgroundRenderResult:
    mask_image = build_text_mask_image(
        text_blocks,
        page_image.size,
        page_rect,
        padding_px=max(0, options.inpaint_padding_px),
    )
    mask_array = np.array(mask_image, dtype=np.uint8)
    if np.count_nonzero(mask_array) == 0:
        logger.info("No overlay mask pixels were generated for background rendering")
        return BackgroundRenderResult(
            image=page_image.convert("RGB").copy(),
            engine_name=None,
            note="No overlay mask pixels were generated.",
            mask_image=mask_image,
        )

    engine, note = resolve_background_inpainting_engine(page_image, mask_image, options)
    fallback_engine = (
        OpenCvFastInpaintingEngine()
        if mask_area_ratio(mask_array) <= options.inpaint_max_area_ratio
        else WhiteBoxInpaintingEngine()
    )
    logger.info("Using background engine %s", engine.name)
    logger.debug("Background engine note: %s", note)
    if engine.name == fallback_engine.name:
        return BackgroundRenderResult(
            image=engine.inpaint(page_image, mask_image),
            engine_name=engine.name,
            note=note,
            mask_image=mask_image,
        )

    try:
        rendered_image = engine.inpaint(page_image, mask_image)
        rendered_note = note
    except BackgroundInpaintingError as error:
        logger.warning(
            "Background engine %s failed; falling back to %s: %s",
            engine.name,
            fallback_engine.name,
            error,
        )
        rendered_image = fallback_engine.inpaint(page_image, mask_image)
        rendered_note = f"{note} Fallback to {fallback_engine.name}: {error}"
    return BackgroundRenderResult(
        image=rendered_image,
        engine_name=engine.name if rendered_note == note else fallback_engine.name,
        note=rendered_note,
        mask_image=mask_image,
    )


def resolve_background_inpainting_engine(
    page_image: Image.Image,
    mask_image: Image.Image,
    options: ConversionOptions,
) -> tuple[BackgroundInpaintingEngine, str]:
    requested_engine = options.inpaint_engine
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    mask_ratio = mask_area_ratio(mask_array)
    diffusion_engine = DiffusionLocalInpaintingEngine(
        command=options.diffusion_command,
        model=options.diffusion_model,
        device=options.diffusion_device,
        max_crop_edge=options.diffusion_max_crop_edge,
        crop_padding_px=max(24, options.inpaint_padding_px * 3),
        timeout_sec=options.diffusion_timeout_sec,
    )
    if requested_engine == "white-box":
        return WhiteBoxInpaintingEngine(), f"Selected white-box engine explicitly (mask area ratio {mask_ratio:.4f})."
    if requested_engine == "opencv-fast":
        return OpenCvFastInpaintingEngine(), (
            f"Selected opencv-fast engine explicitly (mask area ratio {mask_ratio:.4f})."
        )
    if requested_engine == "diffusion-local":
        return diffusion_engine, (
            f"Selected diffusion-local engine explicitly (mask area ratio {mask_ratio:.4f}, "
            f"model {options.diffusion_model}, device {options.diffusion_device})."
        )

    if mask_ratio > options.inpaint_max_area_ratio:
        return WhiteBoxInpaintingEngine(), (
            f"Auto route fell back to white-box because mask area ratio {mask_ratio:.4f} "
            f"exceeded threshold {options.inpaint_max_area_ratio:.4f}."
        )
    complexity = estimate_background_complexity(page_image, mask_image)
    backend_available = shutil.which(options.diffusion_command) is not None
    if backend_available and complexity >= options.diffusion_complexity_threshold:
        return diffusion_engine, (
            f"Auto route selected diffusion-local because complexity score {complexity:.4f} "
            f"met threshold {options.diffusion_complexity_threshold:.4f}."
        )
    if not backend_available and complexity >= options.diffusion_complexity_threshold:
        return OpenCvFastInpaintingEngine(), (
            f"Auto route fell back to opencv-fast because complexity score {complexity:.4f} "
            f"met threshold {options.diffusion_complexity_threshold:.4f} but diffusion command "
            f"'{options.diffusion_command}' was unavailable."
        )
    return OpenCvFastInpaintingEngine(), (
        f"Auto route selected opencv-fast because complexity score {complexity:.4f} "
        f"was below threshold {options.diffusion_complexity_threshold:.4f}."
    )


__all__ = [
    "render_overlay_background",
    "resolve_background_inpainting_engine",
]
