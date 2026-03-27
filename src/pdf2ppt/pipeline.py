from __future__ import annotations

import json
import logging
from typing import Any, Callable

import fitz
from pptx import Presentation

from .background import (
    BackgroundInpaintingEngine,
    BackgroundInpaintingError,
    BackgroundRenderResult,
    DiffusionLocalInpaintingEngine,
    OpenCvFastInpaintingEngine,
    WhiteBoxInpaintingEngine,
    build_mask_shapes,
    build_text_mask_image,
    compute_mask_crop_box,
    estimate_background_complexity,
    invoke_diffusion_backend,
    mask_area_ratio,
    mask_text_regions_with_white_boxes,
    render_overlay_background,
    resolve_background_inpainting_engine,
    write_debug_artifacts,
)
from .block_analysis import (
    bbox_area,
    choose_background_mode,
    classify_page,
    compute_page_signals,
    enrich_ocr_blocks,
    intersection_ratio,
    map_blocks_to_page_coordinates,
    resolve_render_dpi,
    safe_crop,
    score_page,
    select_text_blocks,
)
from .core import (
    ConversionOptions,
    OcrInitializationError,
    OcrPageData,
    OcrProcessingError,
    PageSignals,
)
from .models import ConversionReport, ImagePlacement, PageResult, QualityScore, TextBlock
from .native_extraction import (
    assign_block_roles,
    extract_image_elements,
    extract_native_text_blocks,
    most_common_or_none,
    pil_to_png_bytes,
    promote_ocr_bold_blocks,
    sort_text_blocks,
)
from .ocr import OcrEngine
from .page_analysis import render_page_image
from .ppt_render import (
    add_text_block,
    bbox_to_shape_geometry,
    fit_text_frame,
    pt_to_emu,
    render_page_to_slide,
    resolve_ocr_fit_max_size,
    should_wrap_text_block,
)
from .text_style import (
    build_text_fit_debug_entry,
    choose_measurement_font,
    classify_text_script,
    default_font_family,
    estimate_font_size,
    estimate_text_bold,
    estimate_text_color,
    extract_text_foreground_mask,
    is_cjk,
    measure_text_dimensions,
    script_target_height_ratio,
    script_target_width_ratio,
    single_line_fit_width_ratio,
    write_text_fit_debug_report,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


def convert_pdf(
    options: ConversionOptions,
    *,
    progress_callback: ProgressCallback | None = None,
) -> ConversionReport:
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting PDF conversion: %s -> %s", options.input_path, options.output_path)
    ocr_engine = OcrEngine(
        lang=options.lang,
        use_doc_unwarping=options.use_doc_unwarping,
        det_thresh=options.ocr_det_thresh,
        det_box_thresh=options.ocr_det_box_thresh,
        drop_score=options.ocr_drop_score,
    )
    page_results: list[PageResult] = []

    with fitz.open(options.input_path) as document:
        presentation = Presentation()
        first_page = document[0]
        presentation.slide_width = pt_to_emu(first_page.rect.width)
        presentation.slide_height = pt_to_emu(first_page.rect.height)
        blank_layout = presentation.slide_layouts[6]
        total_pages = document.page_count
        if progress_callback is not None:
            progress_callback(0, total_pages)

        for page_index in range(total_pages):
            page = document[page_index]
            logger.info("Processing page %s/%s", page_index + 1, total_pages)
            try:
                page_result = analyze_page(page, options, ocr_engine)
            except Exception as error:
                logger.exception("Failed to process page %s", page.number + 1)
                raise RuntimeError(f"Failed to process page {page.number + 1}: {error}") from error
            page_results.append(page_result)
            render_page_to_slide(
                presentation,
                blank_layout,
                page_result,
                slide_width_pt=first_page.rect.width,
                slide_height_pt=first_page.rect.height,
            )
            if progress_callback is not None:
                progress_callback(page_index + 1, total_pages)

        presentation.save(options.output_path)

    report = ConversionReport(
        input_path=str(options.input_path),
        output_path=str(options.output_path),
        pages=page_results,
    )
    options.report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Finished PDF conversion with %s page(s)", len(page_results))
    return report


def analyze_page(page: fitz.Page, options: ConversionOptions, ocr_engine: OcrEngine) -> PageResult:
    page_image = render_page_image(page, dpi=resolve_render_dpi(options))
    ocr_reference_image = page_image
    native_blocks, image_boxes = extract_native_text_blocks(page)
    signals = compute_page_signals(page, native_blocks, image_boxes)
    page_kind = classify_page(signals)

    need_ocr = page_kind in {"scanned", "hybrid"} or not native_blocks
    ocr_blocks: list[TextBlock] = []
    if need_ocr:
        ocr_page_data = ocr_engine.extract_text_blocks(page_image, page.number + 1)
        ocr_reference_image = ocr_page_data.image
        ocr_blocks = enrich_ocr_blocks(ocr_page_data.blocks, ocr_reference_image)
        ocr_blocks = map_blocks_to_page_coordinates(ocr_blocks, ocr_reference_image.size, page.rect)

    text_blocks = select_text_blocks(page_kind, native_blocks, ocr_blocks)
    quality = score_page(page_kind, text_blocks, native_blocks, ocr_blocks)
    background_mode, fallback_reason = choose_background_mode(
        page_kind=page_kind,
        quality=quality,
        has_text=bool(text_blocks),
        has_visuals=bool(image_boxes) or signals.drawing_count > 0,
    )
    logger.debug(
        "Page %s classified as %s with background mode %s",
        page.number + 1,
        page_kind,
        background_mode,
    )

    background_png: bytes | None = None
    image_elements: list[ImagePlacement] = []
    background_inpaint_engine: str | None = None
    background_inpaint_note: str | None = None
    mask_image: Any | None = None
    background_image = ocr_reference_image if need_ocr else page_image
    if background_mode == "elements":
        image_elements = extract_image_elements(page, image_boxes, options.render_dpi)
    else:
        if background_mode == "overlay" and text_blocks:
            mask_blocks = [block for block in text_blocks if block.source == "ocr"] or text_blocks
            background_result = render_overlay_background(
                background_image,
                mask_blocks,
                page.rect,
                options=options,
            )
            background_image = background_result.image
            background_inpaint_engine = background_result.engine_name
            background_inpaint_note = background_result.note
            mask_image = background_result.mask_image
        background_png = pil_to_png_bytes(background_image)

    if options.debug_dir is not None and ocr_blocks:
        debug_blocks = [block for block in text_blocks if block.source == "ocr"] or ocr_blocks
        write_debug_artifacts(
            debug_dir=options.debug_dir,
            page_number=page.number + 1,
            page_image=ocr_reference_image,
            masked_image=background_image if background_mode == "overlay" else ocr_reference_image,
            mask_image=mask_image,
            text_blocks=debug_blocks,
            page_rect=page.rect,
            engine_name=background_inpaint_engine,
            engine_note=background_inpaint_note,
        )
        write_text_fit_debug_report(
            debug_dir=options.debug_dir,
            page_number=page.number + 1,
            text_blocks=debug_blocks,
        )

    return PageResult(
        page_number=page.number + 1,
        page_kind=page_kind,
        background_mode=background_mode,
        width_pt=page.rect.width,
        height_pt=page.rect.height,
        text_blocks=text_blocks if background_mode != "full-page" else [],
        quality_score=quality,
        fallback_reason=fallback_reason,
        background_inpaint_engine=background_inpaint_engine,
        background_inpaint_note=background_inpaint_note,
        background_png=background_png,
        image_elements=image_elements,
    )


__all__ = [
    "BackgroundInpaintingEngine",
    "BackgroundInpaintingError",
    "BackgroundRenderResult",
    "ConversionOptions",
    "ConversionReport",
    "DiffusionLocalInpaintingEngine",
    "ImagePlacement",
    "OcrEngine",
    "OcrInitializationError",
    "OcrPageData",
    "OcrProcessingError",
    "OpenCvFastInpaintingEngine",
    "PageResult",
    "PageSignals",
    "ProgressCallback",
    "QualityScore",
    "TextBlock",
    "WhiteBoxInpaintingEngine",
    "add_text_block",
    "analyze_page",
    "assign_block_roles",
    "bbox_area",
    "bbox_to_shape_geometry",
    "build_mask_shapes",
    "build_text_fit_debug_entry",
    "build_text_mask_image",
    "choose_background_mode",
    "choose_measurement_font",
    "classify_page",
    "classify_text_script",
    "compute_mask_crop_box",
    "compute_page_signals",
    "convert_pdf",
    "default_font_family",
    "enrich_ocr_blocks",
    "estimate_background_complexity",
    "estimate_font_size",
    "estimate_text_bold",
    "estimate_text_color",
    "extract_image_elements",
    "extract_native_text_blocks",
    "extract_text_foreground_mask",
    "fit_text_frame",
    "intersection_ratio",
    "invoke_diffusion_backend",
    "is_cjk",
    "map_blocks_to_page_coordinates",
    "mask_area_ratio",
    "mask_text_regions_with_white_boxes",
    "measure_text_dimensions",
    "most_common_or_none",
    "pil_to_png_bytes",
    "promote_ocr_bold_blocks",
    "pt_to_emu",
    "render_overlay_background",
    "render_page_image",
    "render_page_to_slide",
    "resolve_background_inpainting_engine",
    "resolve_ocr_fit_max_size",
    "resolve_render_dpi",
    "safe_crop",
    "score_page",
    "script_target_height_ratio",
    "script_target_width_ratio",
    "select_text_blocks",
    "should_wrap_text_block",
    "single_line_fit_width_ratio",
    "sort_text_blocks",
    "write_debug_artifacts",
    "write_text_fit_debug_report",
]
