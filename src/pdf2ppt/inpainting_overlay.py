from __future__ import annotations

import cv2
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
    compute_mask_crop_box,
    estimate_background_complexity,
    estimate_low_texture_mask_fraction,
    mask_area_ratio,
)
from .models import TextBlock

logger = logging.getLogger(__name__)


AUTO_OPENCV_LARGE_MASK_RATIO_MULTIPLIER = 2.5
AUTO_OPENCV_LARGE_MASK_RATIO_CAP = 0.35
AUTO_OPENCV_LOW_TEXTURE_FRACTION_THRESHOLD = 0.75
TARGETED_FILE_BACK_KEYWORD = "file-back"
TARGETED_FILE_BACK_PRIMARY_TEXT = "歸檔(File-Back)"
TARGETED_FILE_BACK_CONTEXT_GAP_PX = 6
TARGETED_FILE_BACK_CONTEXT_DILATE_PX = 10
TARGETED_FILE_BACK_CROP_PADDING_PX = 24
TARGETED_FILE_BACK_LUMA_MAX_DELTA = 18.0
TARGETED_FILE_BACK_CHROMA_MAX_DELTA = 8.0
TARGETED_FILE_BACK_BLEND_SIGMA = 1.6
TARGETED_FILE_BACK_CONTRAST_SIGMA = 3.0
TARGETED_FILE_BACK_LUMA_SPAN_MAX_SCALE = 1.4
TARGETED_FILE_BACK_LUMA_DETAIL_MAX_SCALE = 2.5


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

    def finalize_result(
        rendered_image: Image.Image,
        *,
        rendered_engine_name: str | None,
        rendered_note: str | None,
    ) -> BackgroundRenderResult:
        corrected_image, debug_images, correction_note = _apply_targeted_file_back_color_correction(
            page_image,
            rendered_image,
            text_blocks,
            page_rect,
            options=options,
        )
        final_note = rendered_note
        if correction_note:
            final_note = f"{rendered_note} {correction_note}" if rendered_note else correction_note
        return BackgroundRenderResult(
            image=corrected_image,
            engine_name=rendered_engine_name,
            note=final_note,
            mask_image=mask_image,
            debug_images=debug_images,
        )

    logger.info("Using background engine %s", engine.name)
    logger.debug("Background engine note: %s", note)
    if engine.name == fallback_engine.name:
        return finalize_result(
            engine.inpaint(page_image, mask_image),
            rendered_engine_name=engine.name,
            rendered_note=note,
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
    return finalize_result(
        rendered_image,
        rendered_engine_name=engine.name if rendered_note == note else fallback_engine.name,
        rendered_note=rendered_note,
    )


def _apply_targeted_file_back_color_correction(
    page_image: Image.Image,
    repaired_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
    *,
    options: ConversionOptions,
) -> tuple[Image.Image, dict[str, Image.Image], str | None]:
    target_blocks = [block for block in text_blocks if block.text == TARGETED_FILE_BACK_PRIMARY_TEXT]
    if not target_blocks:
        target_blocks = [block for block in text_blocks if TARGETED_FILE_BACK_KEYWORD in block.text.lower()]
    if not target_blocks:
        return repaired_image.convert("RGB"), {}, None

    source_rgb = np.array(page_image.convert("RGB"), dtype=np.uint8)
    corrected_rgb = np.array(repaired_image.convert("RGB"), dtype=np.uint8, copy=True)
    debug_images: dict[str, Image.Image] = {}
    notes: list[str] = []

    for index, block in enumerate(target_blocks, start=1):
        block_mask_image = build_text_mask_image(
            [block],
            page_image.size,
            page_rect,
            padding_px=max(0, options.inpaint_padding_px),
        )
        crop_box = compute_mask_crop_box(block_mask_image, padding_px=TARGETED_FILE_BACK_CROP_PADDING_PX)
        if crop_box is None:
            continue

        block_mask = np.array(block_mask_image.convert("L"), dtype=np.uint8) > 0
        if not np.any(block_mask):
            continue

        gap_kernel = np.ones(
            (TARGETED_FILE_BACK_CONTEXT_GAP_PX * 2 + 1, TARGETED_FILE_BACK_CONTEXT_GAP_PX * 2 + 1),
            dtype=np.uint8,
        )
        context_kernel = np.ones(
            (TARGETED_FILE_BACK_CONTEXT_DILATE_PX * 2 + 1, TARGETED_FILE_BACK_CONTEXT_DILATE_PX * 2 + 1),
            dtype=np.uint8,
        )
        inner = cv2.dilate(block_mask.astype(np.uint8), gap_kernel, iterations=1)
        outer = cv2.dilate(inner, context_kernel, iterations=1)
        ring_mask = (outer > 0) & (inner == 0)
        if int(np.count_nonzero(ring_mask)) < 64:
            continue

        x0, y0, x1, y1 = crop_box
        local_component = block_mask[y0:y1, x0:x1]
        local_ring = ring_mask[y0:y1, x0:x1]
        if not np.any(local_component) or not np.any(local_ring):
            continue

        source_crop = source_rgb[y0:y1, x0:x1]
        repaired_crop = corrected_rgb[y0:y1, x0:x1].copy()
        corrected_crop, delta_l, delta_a, delta_b, span_scale, detail_scale = _align_local_luminance_and_chroma_to_ring(
            source_crop,
            repaired_crop,
            local_component,
            local_ring,
            luma_max_delta=TARGETED_FILE_BACK_LUMA_MAX_DELTA,
            chroma_max_delta=TARGETED_FILE_BACK_CHROMA_MAX_DELTA,
            blend_sigma=TARGETED_FILE_BACK_BLEND_SIGMA,
            contrast_sigma=TARGETED_FILE_BACK_CONTRAST_SIGMA,
            luma_span_max_scale=TARGETED_FILE_BACK_LUMA_SPAN_MAX_SCALE,
            luma_detail_max_scale=TARGETED_FILE_BACK_LUMA_DETAIL_MAX_SCALE,
        )
        corrected_rgb[y0:y1, x0:x1] = corrected_crop

        suffix = "file_back" if len(target_blocks) == 1 else f"file_back_{index:02d}"
        debug_images[f"{suffix}_original"] = Image.fromarray(source_crop, mode="RGB")
        debug_images[f"{suffix}_repaired"] = Image.fromarray(repaired_crop, mode="RGB")
        debug_images[f"{suffix}_corrected"] = Image.fromarray(corrected_crop, mode="RGB")
        debug_images[f"{suffix}_mask"] = Image.fromarray((local_component.astype(np.uint8) * 255), mode="L")
        notes.append(
            f"Applied targeted File-Back LAB correction (dL {delta_l:.1f}, dA {delta_a:.1f}, dB {delta_b:.1f}, "
            f"L-span x{span_scale:.2f}, detail x{detail_scale:.2f})."
        )

    if not notes:
        return repaired_image.convert("RGB"), {}, None
    return Image.fromarray(corrected_rgb, mode="RGB"), debug_images, " ".join(notes)


def _align_local_luminance_and_chroma_to_ring(
    source_crop: np.ndarray,
    repaired_crop: np.ndarray,
    local_component: np.ndarray,
    local_ring: np.ndarray,
    *,
    luma_max_delta: float,
    chroma_max_delta: float,
    blend_sigma: float,
    contrast_sigma: float,
    luma_span_max_scale: float,
    luma_detail_max_scale: float,
) -> tuple[np.ndarray, float, float, float, float, float]:
    source_lab = cv2.cvtColor(source_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    repaired_lab = cv2.cvtColor(repaired_crop, cv2.COLOR_RGB2LAB).astype(np.float32)

    ring_l = _trimmed_mean(source_lab[:, :, 0][local_ring])
    ring_a = _trimmed_mean(source_lab[:, :, 1][local_ring])
    ring_b = _trimmed_mean(source_lab[:, :, 2][local_ring])
    component_l = _trimmed_mean(repaired_lab[:, :, 0][local_component])
    component_a = _trimmed_mean(repaired_lab[:, :, 1][local_component])
    component_b = _trimmed_mean(repaired_lab[:, :, 2][local_component])

    delta_l = float(np.clip(ring_l - component_l, -luma_max_delta, luma_max_delta))
    delta_a = float(np.clip(ring_a - component_a, -chroma_max_delta, chroma_max_delta))
    delta_b = float(np.clip(ring_b - component_b, -chroma_max_delta, chroma_max_delta))

    alpha = cv2.GaussianBlur(local_component.astype(np.float32), (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma)
    alpha = np.clip(alpha * 1.5, 0.0, 1.0)[..., None]
    corrected_lab = repaired_lab.copy()
    corrected_lab += alpha * np.array([delta_l, delta_a, delta_b], dtype=np.float32)
    corrected_lab[:, :, 0], span_scale, detail_scale = _restore_local_luminance_contrast(
        source_lab[:, :, 0],
        corrected_lab[:, :, 0],
        local_component,
        local_ring,
        alpha[:, :, 0],
        contrast_sigma=contrast_sigma,
        luma_span_max_scale=luma_span_max_scale,
        luma_detail_max_scale=luma_detail_max_scale,
    )
    corrected_rgb = cv2.cvtColor(np.clip(corrected_lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return corrected_rgb, delta_l, delta_a, delta_b, span_scale, detail_scale


def _restore_local_luminance_contrast(
    source_l: np.ndarray,
    corrected_l: np.ndarray,
    local_component: np.ndarray,
    local_ring: np.ndarray,
    alpha: np.ndarray,
    *,
    contrast_sigma: float,
    luma_span_max_scale: float,
    luma_detail_max_scale: float,
) -> tuple[np.ndarray, float, float]:
    if not np.any(local_component) or not np.any(local_ring):
        return corrected_l, 1.0, 1.0

    ring_low, ring_mid, ring_high = (float(value) for value in np.percentile(source_l[local_ring], [10.0, 50.0, 90.0]))
    component_low, component_mid, component_high = (
        float(value) for value in np.percentile(corrected_l[local_component], [10.0, 50.0, 90.0])
    )
    component_span = max(1.0, component_high - component_low)
    ring_span = max(1.0, ring_high - ring_low)
    span_scale = max(1.0, min(luma_span_max_scale, ring_span / component_span))

    candidate_l = corrected_l.copy()
    candidate_l = component_mid + (candidate_l - component_mid) * span_scale

    source_blur = cv2.GaussianBlur(source_l, (0, 0), sigmaX=contrast_sigma, sigmaY=contrast_sigma)
    ring_detail_std = float((source_l - source_blur)[local_ring].std())
    candidate_blur = cv2.GaussianBlur(candidate_l, (0, 0), sigmaX=contrast_sigma, sigmaY=contrast_sigma)
    candidate_detail = candidate_l - candidate_blur
    component_detail_std = float(candidate_detail[local_component].std())
    detail_scale = min(luma_detail_max_scale, ring_detail_std / max(component_detail_std, 1e-3))
    candidate_l = candidate_l + (detail_scale - 1.0) * candidate_detail

    restored_l = corrected_l * (1.0 - alpha) + candidate_l * alpha
    return np.clip(restored_l, 0.0, 255.0), float(span_scale), float(detail_scale)


def _trimmed_mean(values: np.ndarray, *, trim_ratio: float = 0.1) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values.astype(np.float32, copy=False))
    trim = int(sorted_values.size * trim_ratio)
    if trim > 0 and sorted_values.size > trim * 2:
        sorted_values = sorted_values[trim:-trim]
    return float(np.mean(sorted_values))


def _trimmed_percentiles(
    values: np.ndarray,
    *,
    percentiles: tuple[float, ...],
    trim_ratio: float = 0.1,
) -> tuple[float, ...]:
    if values.size == 0:
        return tuple(0.0 for _ in percentiles)
    sorted_values = np.sort(values.astype(np.float32, copy=False))
    trim = int(sorted_values.size * trim_ratio)
    if trim > 0 and sorted_values.size > trim * 2:
        sorted_values = sorted_values[trim:-trim]
    return tuple(float(np.percentile(sorted_values, percentile)) for percentile in percentiles)


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
        large_mask_opencv_limit = min(
            AUTO_OPENCV_LARGE_MASK_RATIO_CAP,
            options.inpaint_max_area_ratio * AUTO_OPENCV_LARGE_MASK_RATIO_MULTIPLIER,
        )
        low_texture_fraction = estimate_low_texture_mask_fraction(page_image, mask_image)
        if mask_ratio <= large_mask_opencv_limit and (
            low_texture_fraction >= AUTO_OPENCV_LOW_TEXTURE_FRACTION_THRESHOLD
        ):
            return OpenCvFastInpaintingEngine(), (
                f"Auto route selected opencv-fast for a large mask because low-texture mask fraction "
                f"{low_texture_fraction:.4f} met threshold {AUTO_OPENCV_LOW_TEXTURE_FRACTION_THRESHOLD:.4f} "
                f"at mask area ratio {mask_ratio:.4f} (large-mask opencv limit {large_mask_opencv_limit:.4f})."
            )
        return WhiteBoxInpaintingEngine(), (
            f"Auto route fell back to white-box because mask area ratio {mask_ratio:.4f} "
            f"exceeded threshold {options.inpaint_max_area_ratio:.4f}, while low-texture mask fraction "
            f"{low_texture_fraction:.4f} did not justify opencv-fast (large-mask opencv limit "
            f"{large_mask_opencv_limit:.4f}, low-texture threshold "
            f"{AUTO_OPENCV_LOW_TEXTURE_FRACTION_THRESHOLD:.4f})."
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
