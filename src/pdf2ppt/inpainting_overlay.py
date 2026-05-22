from __future__ import annotations

import cv2
import logging
import math

import fitz
import numpy as np
from PIL import Image

from .core import ConversionOptions
from .inpainting_engines import (
    DEFAULT_LAMA_MASK_EXTRA_PADDING_PX,
    BackgroundInpaintingEngine,
    BackgroundInpaintingError,
    BackgroundRenderResult,
    LamaOnnxCudaInpaintingEngine,
    LamaPytorchInpaintingEngine,
    OpenCvFastInpaintingEngine,
    WhiteBoxInpaintingEngine,
    base_lama_inpaint_engine,
    uses_lama_inpaint_engine,
)
from .inpainting_masks import (
    build_refined_text_mask_for_inpainting,
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
TARGETED_FOOTER_LABEL_CONTEXT_GAP_PX = 2
TARGETED_FOOTER_LABEL_CONTEXT_DILATE_PX = 8
TARGETED_FOOTER_LABEL_BROAD_CONTEXT_DILATE_PX = 24
TARGETED_FOOTER_LABEL_CROP_PADDING_PX = 20
TARGETED_FOOTER_LABEL_BLEND_SIGMA = 1.4
TARGETED_FOOTER_LABEL_BLEND_GAIN = 1.6
TARGETED_FOOTER_LABEL_MAX_BROAD_LIFT = 14.0
TARGETED_FOOTER_LABEL_BROAD_TARGET_PERCENTILE = 85.0
FOOTER_BLEED_RESTORE_SOURCE_MIN_GRAY = 160
FOOTER_BLEED_RESTORE_RENDERED_DELTA = 18
FOOTER_BLEED_RESTORE_MIN_RATIO = 0.45
FOOTER_BLEED_RESTORE_OUTER_MIN_RATIO = 0.3
FOOTER_TOP_WASH_SOURCE_MIN_GRAY = 220
FOOTER_TOP_WASH_RENDERED_DELTA = 6
FOOTER_TOP_WASH_MIN_RATIO = 0.18
FOOTER_TOP_WASH_MIN_RUN_RATIO = 0.15


def render_overlay_background(
    page_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
    *,
    options: ConversionOptions,
) -> BackgroundRenderResult:
    mask_padding_px = max(0, options.inpaint_padding_px)
    if uses_lama_inpaint_engine(options.inpaint_engine):
        mask_padding_px += DEFAULT_LAMA_MASK_EXTRA_PADDING_PX
    mask_refinement = build_refined_text_mask_for_inpainting(
        page_image,
        text_blocks,
        page_image.size,
        page_rect,
        padding_px=mask_padding_px,
    )
    mask_image = mask_refinement.mask_image
    mask_array = np.array(mask_image, dtype=np.uint8)
    if np.count_nonzero(mask_array) == 0:
        logger.info("No overlay mask pixels were generated for background rendering")
        return BackgroundRenderResult(
            image=page_image.convert("RGB").copy(),
            engine_name=None,
            note="No overlay mask pixels were generated.",
            mask_image=mask_image,
        )

    protected_line_mask = _build_protected_table_line_mask(mask_refinement, mask_array)
    prepared_page_image = page_image
    if np.any(protected_line_mask):
        prepared_page_image = _neutralize_protected_table_lines(
            page_image,
            mask_array,
            protected_line_mask,
        )

    engine, note = resolve_background_inpainting_engine(page_image, mask_image, options)
    fallback_engine = (
        OpenCvFastInpaintingEngine()
        if mask_area_ratio(mask_array) <= options.inpaint_max_area_ratio
        else WhiteBoxInpaintingEngine()
    )
    if isinstance(engine, OpenCvFastInpaintingEngine):
        engine.set_protected_line_mask(protected_line_mask)
    if isinstance(fallback_engine, OpenCvFastInpaintingEngine):
        fallback_engine.set_protected_line_mask(protected_line_mask)

    def finalize_result(
        rendered_image: Image.Image,
        *,
        rendered_engine_name: str | None,
        rendered_note: str | None,
    ) -> BackgroundRenderResult:
        mask_debug_images = {
            "raw_mask": mask_refinement.raw_mask_image,
            "refined_mask": mask_refinement.refined_mask_image,
        }
        if mask_refinement.table_line_mask_image is not None:
            mask_debug_images["table_line_mask"] = mask_refinement.table_line_mask_image
        if getattr(mask_refinement, "grid_line_mask_image", None) is not None:
            mask_debug_images["grid_line_mask"] = mask_refinement.grid_line_mask_image
        if np.any(protected_line_mask):
            mask_debug_images["protected_line_mask"] = Image.fromarray(
                (protected_line_mask.astype(np.uint8) * 255),
                mode="L",
            )
        engine_debug_note = getattr(engine, "last_debug_note", None)
        engine_debug_images = getattr(engine, "last_debug_images", {})
        if engine_debug_note:
            rendered_note = f"{rendered_note} {engine_debug_note}" if rendered_note else engine_debug_note
        if np.any(protected_line_mask):
            rendered_image = _restore_protected_table_lines(
                page_image,
                rendered_image,
                protected_line_mask,
                text_blocks=text_blocks,
                page_rect=page_rect,
            )
        corrected_image, footer_debug_images, footer_note = _apply_targeted_footer_label_color_correction(
            page_image,
            rendered_image,
            text_blocks,
            page_rect,
            options=options,
        )
        corrected_image, correction_debug_images, correction_note = _apply_targeted_file_back_color_correction(
            page_image,
            corrected_image,
            text_blocks,
            page_rect,
            options=options,
        )
        final_note = rendered_note
        if footer_note:
            final_note = f"{final_note} {footer_note}" if final_note else footer_note
        if correction_note:
            final_note = f"{final_note} {correction_note}" if final_note else correction_note
        return BackgroundRenderResult(
            image=corrected_image,
            engine_name=rendered_engine_name,
            note=final_note,
            mask_image=mask_image,
            debug_images={
                **mask_debug_images,
                **engine_debug_images,
                **footer_debug_images,
                **correction_debug_images,
            },
        )

    logger.info("Using background engine %s", engine.name)
    logger.debug("Background engine note: %s", note)
    if uses_lama_inpaint_engine(options.inpaint_engine):
        return finalize_result(
            engine.inpaint(prepared_page_image, mask_image),
            rendered_engine_name=engine.name,
            rendered_note=note,
        )
    if engine.name == fallback_engine.name:
        return finalize_result(
            engine.inpaint(prepared_page_image, mask_image),
            rendered_engine_name=engine.name,
            rendered_note=note,
        )

    try:
        rendered_image = engine.inpaint(prepared_page_image, mask_image)
        rendered_note = note
    except BackgroundInpaintingError as error:
        logger.warning(
            "Background engine %s failed; falling back to %s: %s",
            engine.name,
            fallback_engine.name,
            error,
        )
        rendered_image = fallback_engine.inpaint(prepared_page_image, mask_image)
        rendered_note = f"{note} Fallback to {fallback_engine.name}: {error}"
    return finalize_result(
        rendered_image,
        rendered_engine_name=engine.name if rendered_note == note else fallback_engine.name,
        rendered_note=rendered_note,
    )


def _build_protected_table_line_mask(mask_refinement: object, mask_array: np.ndarray) -> np.ndarray:
    grid_line_mask_image = getattr(mask_refinement, "grid_line_mask_image", None)
    if grid_line_mask_image is None:
        return np.zeros_like(mask_array, dtype=bool)
    grid_line_mask = np.array(grid_line_mask_image.convert("L"), dtype=np.uint8) > 0
    if not np.any(grid_line_mask):
        return np.zeros_like(mask_array, dtype=bool)
    mask_vicinity = cv2.dilate((mask_array > 0).astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
    return grid_line_mask & mask_vicinity


def _neutralize_protected_table_lines(
    page_image: Image.Image,
    mask_array: np.ndarray,
    protected_line_mask: np.ndarray,
) -> Image.Image:
    source_rgb = np.array(page_image.convert("RGB"), dtype=np.uint8)
    sanitized = source_rgb.copy()
    component_mask = protected_line_mask.astype(np.uint8)
    component_count, labels, _, _ = cv2.connectedComponentsWithStats(component_mask, 8)
    exclusion_mask = (mask_array > 0) | protected_line_mask
    for component_index in range(1, component_count):
        component = labels == component_index
        points = cv2.findNonZero(component.astype(np.uint8))
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        padding = 8
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(source_rgb.shape[1], x + width + padding)
        y1 = min(source_rgb.shape[0], y + height + padding)
        local_component = component[y0:y1, x0:x1]
        local_exclusion = exclusion_mask[y0:y1, x0:x1]
        context_ring = cv2.dilate(local_component.astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1) > 0
        context_ring &= ~local_exclusion
        if not np.any(context_ring):
            continue
        fill_color = np.mean(source_rgb[y0:y1, x0:x1][context_ring], axis=0)
        sanitized[y0:y1, x0:x1][local_component] = fill_color
    return Image.fromarray(sanitized, mode="RGB")


def _restore_protected_table_lines(
    page_image: Image.Image,
    rendered_image: Image.Image,
    protected_line_mask: np.ndarray,
    *,
    text_blocks: list[TextBlock] | None = None,
    page_rect: fitz.Rect | None = None,
) -> Image.Image:
    source_rgb = np.array(page_image.convert("RGB"), dtype=np.uint8)
    rendered_rgb = np.array(rendered_image.convert("RGB"), dtype=np.uint8, copy=True)
    restore_mask = protected_line_mask.copy()
    if np.any(protected_line_mask):
        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
        rendered_gray = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2GRAY)
        line_values = source_gray[protected_line_mask].astype(np.float32)
        gray_cap = float(np.percentile(line_values, 90)) + 80.0
        dilated_mask = cv2.dilate(protected_line_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
        fringe_mask = dilated_mask & (~protected_line_mask)
        downward_seed = cv2.dilate(protected_line_mask.astype(np.uint8), np.ones((1, 5), dtype=np.uint8), iterations=1) > 0
        downward_fringe = np.zeros_like(protected_line_mask, dtype=bool)
        for dy in range(1, 4):
            downward_fringe[dy:, :] |= downward_seed[:-dy, :]
        fringe_mask |= downward_fringe & (~protected_line_mask)
        fringe_mask &= source_gray <= gray_cap
        fringe_mask &= (source_gray.astype(np.int16) + 12) < rendered_gray.astype(np.int16)
        restore_mask |= fringe_mask
    else:
        source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
        rendered_gray = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2GRAY)

    if text_blocks and page_rect is not None:
        restore_mask |= _build_footer_bottom_border_restore_mask(
            source_gray,
            rendered_gray,
            protected_line_mask,
            text_blocks=text_blocks,
            page_rect=page_rect,
        )
    rendered_rgb[restore_mask] = source_rgb[restore_mask]
    return Image.fromarray(rendered_rgb, mode="RGB")


def _build_footer_bottom_border_restore_mask(
    source_gray: np.ndarray,
    rendered_gray: np.ndarray,
    protected_line_mask: np.ndarray,
    *,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
) -> np.ndarray:
    restore_mask = np.zeros_like(protected_line_mask, dtype=bool)
    image_height, image_width = source_gray.shape
    scale_x = image_width / max(1.0, float(page_rect.width))
    scale_y = image_height / max(1.0, float(page_rect.height))

    for block in text_blocks:
        if not _is_footer_border_restore_candidate(block, page_rect=page_rect):
            continue
        width_pt = block.bbox[2] - block.bbox[0]
        height_pt = block.bbox[3] - block.bbox[1]
        if width_pt < 120.0 or height_pt > 42.0:
            continue

        x0 = max(0, int(math.floor(block.bbox[0] * scale_x)) - 2)
        y0 = max(0, int(math.floor(block.bbox[1] * scale_y)))
        x1 = min(image_width, int(math.ceil(block.bbox[2] * scale_x)) + 2)
        y1 = min(image_height, int(math.ceil(block.bbox[3] * scale_y)))
        if x1 <= x0 or y1 <= y0:
            continue

        restore_mask |= _build_footer_dark_bleed_restore_mask(
            source_gray,
            rendered_gray,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
        )
        restore_mask |= _build_footer_top_wash_restore_mask(
            source_gray,
            rendered_gray,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
        )
        restore_mask |= _build_footer_top_outer_wash_restore_mask(
            source_gray,
            rendered_gray,
            x0=x0,
            x1=x1,
            y0=y0,
        )

        seed_y0 = max(0, y1 - 2)
        seed_y1 = min(image_height, y1 + 3)
        seed_region = protected_line_mask[seed_y0:seed_y1, x0:x1]
        if np.count_nonzero(seed_region) < max(4, int((x1 - x0) * 0.02)):
            continue

        seed_values = source_gray[seed_y0:seed_y1, x0:x1][seed_region].astype(np.float32)
        if seed_values.size == 0:
            continue
        gray_cap = float(np.percentile(seed_values, 90)) + 80.0

        for row in range(y1, min(image_height, y1 + 7)):
            source_row = source_gray[row, x0:x1].astype(np.int16)
            rendered_row = rendered_gray[row, x0:x1].astype(np.int16)
            candidate = (source_row <= gray_cap) & ((source_row + 12) < rendered_row)
            if np.count_nonzero(candidate) < int((x1 - x0) * 0.7):
                continue
            candidate_mask = np.zeros_like(protected_line_mask, dtype=bool)
            candidate_mask[row, x0:x1] = candidate
            candidate_mask = cv2.morphologyEx(candidate_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 7), dtype=np.uint8), iterations=1) > 0
            restore_mask |= candidate_mask

    return restore_mask


def _build_footer_dark_bleed_restore_mask(
    source_gray: np.ndarray,
    rendered_gray: np.ndarray,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> np.ndarray:
    restore_mask = np.zeros_like(source_gray, dtype=bool)
    image_height = source_gray.shape[0]
    inner_min_coverage = max(12, int((x1 - x0) * FOOTER_BLEED_RESTORE_MIN_RATIO))
    outer_min_coverage = max(12, int((x1 - x0) * FOOTER_BLEED_RESTORE_OUTER_MIN_RATIO))
    row_ranges = (
        (range(max(y0, y1 - 6), y1), inner_min_coverage),
        (range(y1, min(image_height, y1 + 6)), outer_min_coverage),
    )
    for rows, min_coverage in row_ranges:
        for row in rows:
            source_row = source_gray[row, x0:x1].astype(np.int16)
            rendered_row = rendered_gray[row, x0:x1].astype(np.int16)
            candidate = (source_row >= FOOTER_BLEED_RESTORE_SOURCE_MIN_GRAY) & (
                (rendered_row + FOOTER_BLEED_RESTORE_RENDERED_DELTA) < source_row
            )
            if np.count_nonzero(candidate) < min_coverage:
                continue
            candidate_mask = np.zeros_like(source_gray, dtype=bool)
            candidate_mask[row, x0:x1] = candidate
            candidate_mask = cv2.morphologyEx(candidate_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 7), dtype=np.uint8), iterations=1) > 0
            restore_mask |= candidate_mask
    return restore_mask


def _build_footer_top_wash_restore_mask(
    source_gray: np.ndarray,
    rendered_gray: np.ndarray,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> np.ndarray:
    restore_mask = np.zeros_like(source_gray, dtype=bool)
    max_rows = min(y1, y0 + 4)
    min_coverage = max(24, int((x1 - x0) * FOOTER_TOP_WASH_MIN_RATIO))
    min_run = max(20, int((x1 - x0) * FOOTER_TOP_WASH_MIN_RUN_RATIO))
    for row in range(y0, max_rows):
        source_row = source_gray[row, x0:x1].astype(np.int16)
        rendered_row = rendered_gray[row, x0:x1].astype(np.int16)
        candidate = (source_row >= FOOTER_TOP_WASH_SOURCE_MIN_GRAY) & (
            (rendered_row + FOOTER_TOP_WASH_RENDERED_DELTA) < source_row
        )
        count = int(np.count_nonzero(candidate))
        if count < min_coverage:
            continue
        longest_run = 0
        current_run = 0
        for value in candidate:
            if value:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
        if longest_run < min_run:
            continue
        candidate_mask = np.zeros_like(source_gray, dtype=bool)
        candidate_mask[row, x0:x1] = candidate
        candidate_mask = cv2.morphologyEx(candidate_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 9), dtype=np.uint8), iterations=1) > 0
        restore_mask |= candidate_mask
    return restore_mask


def _build_footer_top_outer_wash_restore_mask(
    source_gray: np.ndarray,
    rendered_gray: np.ndarray,
    *,
    x0: int,
    x1: int,
    y0: int,
) -> np.ndarray:
    restore_mask = np.zeros_like(source_gray, dtype=bool)
    min_coverage = max(24, int((x1 - x0) * FOOTER_TOP_WASH_MIN_RATIO))
    min_run = max(20, int((x1 - x0) * FOOTER_TOP_WASH_MIN_RUN_RATIO))
    for row in range(max(0, y0 - 3), y0):
        source_row = source_gray[row, x0:x1].astype(np.int16)
        rendered_row = rendered_gray[row, x0:x1].astype(np.int16)
        candidate = (source_row >= FOOTER_TOP_WASH_SOURCE_MIN_GRAY) & (
            (rendered_row + FOOTER_TOP_WASH_RENDERED_DELTA) < source_row
        )
        count = int(np.count_nonzero(candidate))
        if count < min_coverage:
            continue
        longest_run = 0
        current_run = 0
        for value in candidate:
            if value:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
        if longest_run < min_run:
            continue
        candidate_mask = np.zeros_like(source_gray, dtype=bool)
        candidate_mask[row, x0:x1] = candidate
        candidate_mask = cv2.morphologyEx(candidate_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 9), dtype=np.uint8), iterations=1) > 0
        restore_mask |= candidate_mask
    return restore_mask


def _is_footer_border_restore_candidate(block: TextBlock, *, page_rect: fitz.Rect) -> bool:
    if block.source != "ocr" or "\n" in block.text:
        return False
    if block.block_role != "body":
        return False
    return block.bbox[3] >= float(page_rect.height) * 0.85


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


def _apply_targeted_footer_label_color_correction(
    page_image: Image.Image,
    repaired_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
    *,
    options: ConversionOptions,
) -> tuple[Image.Image, dict[str, Image.Image], str | None]:
    target_blocks = [block for block in text_blocks if _is_footer_label_color_correction_candidate(block, page_rect=page_rect)]
    if not target_blocks:
        return repaired_image.convert("RGB"), {}, None

    source_rgb = np.array(page_image.convert("RGB"), dtype=np.uint8)
    corrected_rgb = np.array(repaired_image.convert("RGB"), dtype=np.uint8, copy=True)
    debug_images: dict[str, Image.Image] = {}
    notes: list[str] = []
    shared_target_rgb = _resolve_shared_footer_label_target_rgb(
        source_rgb,
        page_image.size,
        target_blocks,
        page_rect,
    )

    for index, block in enumerate(target_blocks, start=1):
        block_mask_image = build_text_mask_image(
            [block],
            page_image.size,
            page_rect,
            padding_px=0,
        )
        crop_box = compute_mask_crop_box(block_mask_image, padding_px=TARGETED_FOOTER_LABEL_CROP_PADDING_PX)
        if crop_box is None:
            continue

        block_mask = np.array(block_mask_image.convert("L"), dtype=np.uint8) > 0
        if not np.any(block_mask):
            continue

        gap_kernel = np.ones(
            (TARGETED_FOOTER_LABEL_CONTEXT_GAP_PX * 2 + 1, TARGETED_FOOTER_LABEL_CONTEXT_GAP_PX * 2 + 1),
            dtype=np.uint8,
        )
        context_kernel = np.ones(
            (TARGETED_FOOTER_LABEL_CONTEXT_DILATE_PX * 2 + 1, TARGETED_FOOTER_LABEL_CONTEXT_DILATE_PX * 2 + 1),
            dtype=np.uint8,
        )
        broad_context_kernel = np.ones(
            (TARGETED_FOOTER_LABEL_BROAD_CONTEXT_DILATE_PX * 2 + 1, TARGETED_FOOTER_LABEL_BROAD_CONTEXT_DILATE_PX * 2 + 1),
            dtype=np.uint8,
        )
        inner = cv2.dilate(block_mask.astype(np.uint8), gap_kernel, iterations=1)
        outer = cv2.dilate(inner, context_kernel, iterations=1)
        ring_mask = (outer > 0) & (inner == 0)
        if int(np.count_nonzero(ring_mask)) < 64:
            continue
        broad_outer = cv2.dilate(inner, broad_context_kernel, iterations=1)
        broad_ring_mask = (broad_outer > 0) & (outer == 0)

        x0, y0, x1, y1 = crop_box
        local_component = block_mask[y0:y1, x0:x1]
        local_ring = ring_mask[y0:y1, x0:x1]
        local_broad_ring = broad_ring_mask[y0:y1, x0:x1]
        if not np.any(local_component) or not np.any(local_ring):
            continue

        source_crop = source_rgb[y0:y1, x0:x1]
        repaired_crop = corrected_rgb[y0:y1, x0:x1].copy()
        target_rgb = shared_target_rgb
        broad_lift = 0.0
        if target_rgb is None:
            target_rgb, broad_lift = _resolve_footer_label_target_rgb(
                source_crop,
                local_ring,
                local_broad_ring,
                max_broad_lift=TARGETED_FOOTER_LABEL_MAX_BROAD_LIFT,
            )
        corrected_crop, delta_rgb = _blend_local_component_toward_ring(
            target_rgb,
            repaired_crop,
            local_component,
            blend_sigma=TARGETED_FOOTER_LABEL_BLEND_SIGMA,
            blend_gain=TARGETED_FOOTER_LABEL_BLEND_GAIN,
        )
        corrected_rgb[y0:y1, x0:x1] = corrected_crop

        suffix = f"footer_label_{index:02d}"
        debug_images[f"{suffix}_corrected"] = Image.fromarray(corrected_crop, mode="RGB")
        debug_images[f"{suffix}_mask"] = Image.fromarray((local_component.astype(np.uint8) * 255), mode="L")
        notes.append(
            f"Applied footer label flat tone correction to '{block.text}' "
            f"(dR {delta_rgb[0]:.1f}, dG {delta_rgb[1]:.1f}, dB {delta_rgb[2]:.1f}, broad lift {broad_lift:.1f})."
        )

    if not notes:
        return repaired_image.convert("RGB"), {}, None
    return Image.fromarray(corrected_rgb, mode="RGB"), debug_images, " ".join(notes)


def _resolve_shared_footer_label_target_rgb(
    source_rgb: np.ndarray,
    image_size: tuple[int, int],
    target_blocks: list[TextBlock],
    page_rect: fitz.Rect,
) -> np.ndarray | None:
    target_samples: list[np.ndarray] = []
    for block in target_blocks:
        block_mask_image = build_text_mask_image(
            [block],
            image_size,
            page_rect,
            padding_px=0,
        )
        crop_box = compute_mask_crop_box(block_mask_image, padding_px=TARGETED_FOOTER_LABEL_CROP_PADDING_PX)
        if crop_box is None:
            continue
        block_mask = np.array(block_mask_image.convert("L"), dtype=np.uint8) > 0
        if not np.any(block_mask):
            continue
        gap_kernel = np.ones(
            (TARGETED_FOOTER_LABEL_CONTEXT_GAP_PX * 2 + 1, TARGETED_FOOTER_LABEL_CONTEXT_GAP_PX * 2 + 1),
            dtype=np.uint8,
        )
        context_kernel = np.ones(
            (TARGETED_FOOTER_LABEL_CONTEXT_DILATE_PX * 2 + 1, TARGETED_FOOTER_LABEL_CONTEXT_DILATE_PX * 2 + 1),
            dtype=np.uint8,
        )
        broad_context_kernel = np.ones(
            (TARGETED_FOOTER_LABEL_BROAD_CONTEXT_DILATE_PX * 2 + 1, TARGETED_FOOTER_LABEL_BROAD_CONTEXT_DILATE_PX * 2 + 1),
            dtype=np.uint8,
        )
        inner = cv2.dilate(block_mask.astype(np.uint8), gap_kernel, iterations=1)
        outer = cv2.dilate(inner, context_kernel, iterations=1)
        broad_outer = cv2.dilate(inner, broad_context_kernel, iterations=1)
        broad_ring_mask = (broad_outer > 0) & (outer == 0)
        x0, y0, x1, y1 = crop_box
        local_broad_ring = broad_ring_mask[y0:y1, x0:x1]
        if not np.any(local_broad_ring):
            continue
        source_crop = source_rgb[y0:y1, x0:x1]
        target_samples.append(
            np.percentile(
                source_crop[local_broad_ring].astype(np.float32),
                TARGETED_FOOTER_LABEL_BROAD_TARGET_PERCENTILE,
                axis=0,
            )
        )
    if not target_samples:
        return None
    return np.mean(np.stack(target_samples, axis=0), axis=0).astype(np.float32)


def _is_footer_label_color_correction_candidate(block: TextBlock, *, page_rect: fitz.Rect) -> bool:
    if block.source != "ocr" or block.block_role != "body" or "\n" in block.text:
        return False
    if "%" not in block.text:
        return False
    width_pt = block.bbox[2] - block.bbox[0]
    height_pt = block.bbox[3] - block.bbox[1]
    if width_pt < 140.0 or width_pt > 240.0 or height_pt > 42.0:
        return False
    return block.bbox[3] >= float(page_rect.height) * 0.85


def _blend_local_component_toward_ring(
    target_rgb: np.ndarray,
    repaired_crop: np.ndarray,
    local_component: np.ndarray,
    *,
    blend_sigma: float,
    blend_gain: float,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    if not np.any(local_component):
        return repaired_crop, (0.0, 0.0, 0.0)
    component_mean = repaired_crop[local_component].astype(np.float32).mean(axis=0)
    alpha = cv2.GaussianBlur(local_component.astype(np.float32), (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma)
    alpha = np.clip(alpha * blend_gain, 0.0, 1.0)[..., None]
    corrected = repaired_crop.astype(np.float32)
    corrected = corrected * (1.0 - alpha) + target_rgb[None, None, :] * alpha
    delta = tuple(float(target_rgb[index] - component_mean[index]) for index in range(3))
    return np.clip(corrected, 0.0, 255.0).astype(np.uint8), delta


def _resolve_footer_label_target_rgb(
    source_crop: np.ndarray,
    local_ring: np.ndarray,
    local_broad_ring: np.ndarray,
    *,
    max_broad_lift: float,
) -> tuple[np.ndarray, float]:
    near_mean = source_crop[local_ring].astype(np.float32).mean(axis=0)
    if not np.any(local_broad_ring):
        return near_mean, 0.0
    broad_target = np.percentile(
        source_crop[local_broad_ring].astype(np.float32),
        TARGETED_FOOTER_LABEL_BROAD_TARGET_PERCENTILE,
        axis=0,
    )
    near_luma = float(np.mean(near_mean))
    broad_luma = float(np.mean(broad_target))
    requested_lift = max(0.0, broad_luma - near_luma)
    if requested_lift <= 0.0:
        return near_mean, 0.0
    applied_lift = min(requested_lift, max_broad_lift)
    blend = applied_lift / requested_lift
    target_rgb = near_mean + (broad_target - near_mean) * blend
    return target_rgb.astype(np.float32), applied_lift


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
    # Auto routing is intentionally conservative:
    # 1. Small/normal masks stay on opencv-fast.
    # 2. Large masks usually fall back to white-box.
    # 3. A large mask may still stay on opencv-fast when most masked pixels sit on
    #    low-texture background and the mask is not too far beyond the configured cap.
    requested_engine = options.inpaint_engine
    mask_array = np.array(mask_image.convert("L"), dtype=np.uint8)
    mask_ratio = mask_area_ratio(mask_array)
    if requested_engine == "white-box":
        return WhiteBoxInpaintingEngine(), f"Selected white-box engine explicitly (mask area ratio {mask_ratio:.4f})."
    if requested_engine == "opencv-fast":
        return OpenCvFastInpaintingEngine(), (
            f"Selected opencv-fast engine explicitly (mask area ratio {mask_ratio:.4f})."
        )
    if base_lama_inpaint_engine(requested_engine) == "lama-onnx-cuda":
        return LamaOnnxCudaInpaintingEngine(
            model_root=options.inpaint_model_root,
            cuda_provider=options.inpaint_onnx_cuda_provider,
            execution_mode=options.inpaint_onnx_execution_mode,
            max_side_px=options.inpaint_max_side_px,
            patch_hybrid=options.inpaint_lama_patch_hybrid,
        ), (
            f"Selected {requested_engine} engine explicitly (mask area ratio {mask_ratio:.4f}, "
            f"provider {options.inpaint_onnx_cuda_provider}, execution_mode {options.inpaint_onnx_execution_mode}, "
            f"patch_hybrid={options.inpaint_lama_patch_hybrid})."
        )
    if base_lama_inpaint_engine(requested_engine) == "lama-pytorch":
        return LamaPytorchInpaintingEngine(
            model_root=options.inpaint_model_root,
            repo_root=options.inpaint_lama_repo_root,
            device=options.inpaint_lama_device,
            python_executable=options.inpaint_lama_python_executable,
            max_side_px=options.inpaint_max_side_px,
            patch_hybrid=options.inpaint_lama_patch_hybrid,
        ), (
            f"Selected {requested_engine} engine explicitly (mask area ratio {mask_ratio:.4f}, "
            f"repo_root {options.inpaint_lama_repo_root}, device {options.inpaint_lama_device}, "
            f"patch_hybrid={options.inpaint_lama_patch_hybrid})."
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
    return OpenCvFastInpaintingEngine(), (
        f"Auto route selected opencv-fast because mask area ratio {mask_ratio:.4f} "
        f"stayed within threshold {options.inpaint_max_area_ratio:.4f} "
        f"(complexity score {complexity:.4f})."
    )


__all__ = [
    "render_overlay_background",
    "resolve_background_inpainting_engine",
]
