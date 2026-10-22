from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import fitz
from PIL import Image
from pptx import Presentation

from .background import (
    BackgroundInpaintingEngine,
    BackgroundInpaintingError,
    BackgroundRenderResult,
    OpenCvFastInpaintingEngine,
    WhiteBoxInpaintingEngine,
    build_mask_shapes,
    build_text_mask_image,
    compute_mask_crop_box,
    estimate_background_complexity,
    mask_area_ratio,
    mask_text_regions_with_white_boxes,
    render_overlay_background,
    resolve_background_inpainting_engine,
    write_debug_artifacts,
)
from .background_erase import (
    erase_regions_from_image,
    paint_events_from_page,
    promote_shapes_under_raster,
    regions_for_background_erase,
    resolve_erase_fill_color,
    uncomposite_translucent_shapes_from_image,
)
from .block_analysis import (
    bbox_area,
    choose_background_mode,
    classify_page,
    compute_page_signals,
    enrich_ocr_blocks,
    filter_suspicious_ocr_blocks,
    intersection_ratio,
    map_blocks_to_page_coordinates,
    resolve_background_render_dpi,
    resolve_render_dpi,
    safe_crop,
    score_page,
    select_text_blocks,
)
from .core import (
    DEFAULT_OCR_BATCH_SIZE,
    ConversionOptions,
    InputValidationError,
    OcrInitializationError,
    OcrPageData,
    OcrProcessingError,
    PageConversionError,
    PageSignals,
)
from .generative_inpaint import (
    GenerativeBudget,
    build_ocr_residual_checker,
    uses_generative_inpaint_engine,
)
from .layout import (
    LayoutEngine,
    apply_layout_to_page,
    extract_figure_image_placements,
    split_wide_multiline_blocks,
)
from .layout_planner import (
    LayoutPlannerBudget,
    LayoutPlannerService,
    build_layout_planner_service,
    uses_layout_planner,
)
from .models import ConversionReport, ImagePlacement, PageResult, QualityScore, TextBlock
from .native_extraction import (
    assign_block_roles,
    extract_image_elements,
    extract_native_text_blocks,
    most_common_or_none,
    pil_to_image_bytes,
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
    resolve_vertical_anchor,
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
from .vector_nativization import (
    apply_cell_fills,
    extract_vector_content,
    filter_text_blocks_outside_tables,
    remaining_visuals_after_nativization,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]

NOTEBOOKLM_WATERMARK_TOKEN = "notebooklm"
NOTEBOOKLM_WATERMARK_RIGHT_MIN_RATIO = 0.72
NOTEBOOKLM_WATERMARK_TOP_MIN_RATIO = 0.9
NOTEBOOKLM_WATERMARK_MAX_WIDTH_RATIO = 0.2
NOTEBOOKLM_WATERMARK_MAX_HEIGHT_RATIO = 0.06
NOTEBOOKLM_WATERMARK_FALLBACK_WIDTH_RATIO = 0.075
NOTEBOOKLM_WATERMARK_FALLBACK_HEIGHT_RATIO = 0.028
NOTEBOOKLM_WATERMARK_FALLBACK_RIGHT_MARGIN_RATIO = 0.008
NOTEBOOKLM_WATERMARK_FALLBACK_BOTTOM_MARGIN_RATIO = 0.008


def convert_pdf(
    options: ConversionOptions,
    *,
    progress_callback: ProgressCallback | None = None,
    write_pptx: bool = True,
    emit_ir_path: Path | None = None,
) -> ConversionReport:
    """Analyze a PDF and optionally write PPTX and/or SlideIR.

    Default behaviour (``write_pptx=True``, ``emit_ir_path=None``) is unchanged
    from pre-P5: produce PPTX + conversion report. Agents can set
    ``write_pptx=False`` with ``emit_ir_path`` to extract SlideIR only.
    """
    conversion_started_at = perf_counter()
    if not options.input_path.exists():
        raise InputValidationError(f"Input PDF does not exist: {options.input_path}")
    if write_pptx:
        options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting PDF conversion: %s -> %s (pptx=%s, ir=%s)",
        options.input_path,
        options.output_path,
        write_pptx,
        emit_ir_path,
    )
    page_results, slide_width_pt, slide_height_pt = analyze_pdf(
        options,
        progress_callback=progress_callback,
    )

    if write_pptx:
        render_pages_to_pptx(
            page_results,
            options.output_path,
            slide_width_pt=slide_width_pt,
            slide_height_pt=slide_height_pt,
        )

    refine_report_payload: dict[str, object] | None = None
    if write_pptx and bool(getattr(options, "refine", False)):
        from .refine import refine_pages, write_refine_report

        refine_iters = int(getattr(options, "refine_iters", 2) or 0)
        page_results, refine_report = refine_pages(
            page_results,
            pdf_path=options.input_path,
            output_pptx=options.output_path,
            slide_width_pt=slide_width_pt,
            slide_height_pt=slide_height_pt,
            iterations=refine_iters,
            max_fixes_per_slide=int(getattr(options, "refine_max_fixes_per_slide", 8)),
            max_attempts_per_slide=int(getattr(options, "refine_max_attempts_per_slide", 8)),
            time_budget_seconds=float(getattr(options, "refine_time_budget_s", 90.0)),
            min_font_size_pt=float(getattr(options, "refine_min_font_size_pt", 8.0)),
            min_font_ratio=float(getattr(options, "refine_min_font_ratio", 0.85)),
            dpi=float(getattr(options, "refine_dpi", 144) or 144),
        )
        refine_report_payload = refine_report.to_dict()
        refine_report_path = getattr(options, "refine_report_path", None)
        if refine_report_path is None:
            refine_report_path = options.output_path.with_suffix(".refine.json")
        write_refine_report(refine_report, Path(refine_report_path))
        logger.info(
            "Refine finished: skipped=%s runtime=%.3fs report=%s",
            refine_report.skipped_reason,
            refine_report.runtime_seconds or 0.0,
            refine_report_path,
        )

    layout_planner_report = None
    layout_planner_budget = getattr(options, "layout_planner_budget", None)
    if isinstance(layout_planner_budget, LayoutPlannerBudget):
        layout_planner_report = layout_planner_budget.to_dict()
        skip_reason = getattr(options, "layout_planner_skip_reason", None)
        if skip_reason:
            layout_planner_report["skip_reason"] = skip_reason
        layout_planner_report["apply_grouping"] = bool(
            getattr(options, "layout_planner_apply_grouping", False)
        )
        layout_planner_report["apply_fonts"] = bool(
            getattr(options, "layout_planner_apply_fonts", False)
        )
        logger.info(
            "Layout planner run stats: calls=%s http_attempts=%s cache_hits=%s fallbacks=%s "
            "circuit_open=%s est_cost_usd=%.4f",
            layout_planner_budget.calls,
            layout_planner_budget.http_attempts,
            layout_planner_budget.cache_hits,
            layout_planner_budget.fallbacks,
            layout_planner_budget.circuit_open,
            layout_planner_budget.estimated_cost_usd,
        )
    elif getattr(options, "layout_planner_skip_reason", None):
        layout_planner_report = {
            "skip_reason": options.layout_planner_skip_reason,
            "calls": 0,
            "http_attempts": 0,
            "cache_hits": 0,
            "fallbacks": 0,
            "estimated_cost_usd": 0.0,
        }

    if emit_ir_path is not None:
        from .ir_io import write_slide_ir

        # Emit after refine so SlideIR matches the final PPTX when both are requested.
        deck_plan_metadata = None
        if layout_planner_report is not None:
            deck_plan_metadata = {
                "engine": "layout_planner",
                "planner": getattr(options, "layout_planner", None),
                "stats": layout_planner_report,
            }
        write_slide_ir(
            page_results,
            emit_ir_path,
            slide_width_pt=slide_width_pt,
            slide_height_pt=slide_height_pt,
            source_path=options.input_path,
            plan_metadata=deck_plan_metadata,
        )

    generative_report = None
    if isinstance(options.generative_budget, GenerativeBudget):
        budget = options.generative_budget
        generative_report = budget.to_dict()
        logger.info(
            "Generative inpaint run stats: calls=%s cache_hits=%s fallbacks=%s est_cost_usd=%.4f",
            budget.calls,
            budget.cache_hits,
            budget.fallbacks,
            budget.estimated_cost_usd,
        )
    report = ConversionReport(
        input_path=str(options.input_path),
        output_path=str(options.output_path) if write_pptx else "",
        pages=page_results,
        generative_inpaint=generative_report,
        refine=refine_report_payload,
        layout_planner=layout_planner_report,
    )
    options.report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Finished PDF conversion with %s page(s) in %.3fs",
        len(page_results),
        perf_counter() - conversion_started_at,
    )
    return report


def analyze_pdf(
    options: ConversionOptions,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[PageResult], float, float]:
    """Run the analysis pipeline and return page results plus slide size in pt."""
    ocr_engine = OcrEngine(
        lang=options.lang,
        model_root=options.ocr_model_root,
        use_doc_orientation=options.ocr_use_doc_orientation,
        use_textline_orientation=options.ocr_use_textline_orientation,
        use_doc_unwarping=options.use_doc_unwarping,
        det_thresh=options.ocr_det_thresh,
        det_box_thresh=options.ocr_det_box_thresh,
        drop_score=options.ocr_drop_score,
        return_word_box=options.ocr_return_word_box,
    )
    layout_engine = LayoutEngine(
        engine=str(getattr(options, "layout_engine", "heuristic")),
        model_root=getattr(options, "layout_model_root", None) or options.ocr_model_root,
        skip_header_footer=bool(getattr(options, "layout_skip_header_footer", True)),
    )
    layout_planner_service: LayoutPlannerService | None = None
    if uses_layout_planner(getattr(options, "layout_planner", None)):
        layout_planner_service = build_layout_planner_service(options)
    if uses_generative_inpaint_engine(options.inpaint_engine):
        if options.generative_budget is None:
            options.generative_budget = GenerativeBudget(
                max_calls=options.generative_max_calls,
                max_cost_usd=options.generative_max_cost_usd,
                max_attempts=options.generative_max_attempts,
            )
        if options.generative_verify_ocr and options.generative_residual_checker is None:
            options.generative_residual_checker = build_ocr_residual_checker(ocr_engine)

    page_results: list[PageResult] = []
    with fitz.open(options.input_path) as document:
        first_page = document[0]
        slide_width_pt = float(first_page.rect.width)
        slide_height_pt = float(first_page.rect.height)
        total_pages = document.page_count
        batched_ocr_page_data = prepare_batched_ocr_page_data(document, options, ocr_engine)
        if progress_callback is not None:
            progress_callback(0, total_pages)

        for page_index in range(total_pages):
            page = document[page_index]
            logger.info("Processing page %s/%s", page_index + 1, total_pages)
            try:
                page_result = analyze_page(
                    page,
                    options,
                    ocr_engine,
                    layout_engine=layout_engine,
                    layout_planner_service=layout_planner_service,
                    precomputed_ocr_page_data=batched_ocr_page_data.get(page.number + 1),
                )
            except (OcrInitializationError, OcrProcessingError, PageConversionError):
                raise
            except Exception as error:
                logger.exception("Failed to process page %s", page.number + 1)
                raise PageConversionError(
                    page.number + 1, f"Failed to process page {page.number + 1}: {error}"
                ) from error
            page_results.append(page_result)
            if progress_callback is not None:
                progress_callback(page_index + 1, total_pages)

    return page_results, slide_width_pt, slide_height_pt


def render_pages_to_pptx(
    pages: list[PageResult],
    output_path: Path,
    *,
    slide_width_pt: float,
    slide_height_pt: float,
) -> Path:
    """Render analyzed (or deserialized) pages to a PPTX file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    presentation = Presentation()
    presentation.slide_width = pt_to_emu(slide_width_pt)
    presentation.slide_height = pt_to_emu(slide_height_pt)
    blank_layout = presentation.slide_layouts[6]
    for page_result in pages:
        slide_render_started_at = perf_counter()
        render_page_to_slide(
            presentation,
            blank_layout,
            page_result,
            slide_width_pt=slide_width_pt,
            slide_height_pt=slide_height_pt,
        )
        logger.info(
            "Page %s slide render: %.3fs",
            page_result.page_number,
            perf_counter() - slide_render_started_at,
        )
    presentation_save_started_at = perf_counter()
    presentation.save(str(output))
    logger.info("Presentation save: %.3fs", perf_counter() - presentation_save_started_at)
    return output


def convert_from_ir(
    ir_path: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
) -> ConversionReport:
    """Load SlideIR JSON and render a PPTX (no PDF required)."""
    from .ir_io import read_slide_ir, render_deck_to_pptx

    ir_file = Path(ir_path)
    out_file = Path(output_path)
    deck = read_slide_ir(ir_file)
    render_deck_to_pptx(deck, out_file)
    report = ConversionReport(
        input_path=str(ir_file),
        output_path=str(out_file),
        pages=deck.pages,
    )
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def analyze_page(
    page: fitz.Page,
    options: ConversionOptions,
    ocr_engine: OcrEngine,
    *,
    layout_engine: LayoutEngine | None = None,
    layout_planner_service: LayoutPlannerService | None = None,
    precomputed_ocr_page_data: tuple[Image.Image, OcrPageData] | None = None,
) -> PageResult:
    page_started_at = perf_counter()
    native_extract_seconds = 0.0
    signal_seconds = 0.0
    ocr_render_seconds = 0.0
    ocr_seconds = 0.0
    ocr_enrich_seconds = 0.0
    page_mode_seconds = 0.0
    background_render_seconds = 0.0
    background_process_seconds = 0.0
    background_encode_seconds = 0.0

    native_extract_started_at = perf_counter()
    native_blocks, image_boxes = extract_native_text_blocks(page)
    native_extract_seconds = perf_counter() - native_extract_started_at
    signal_started_at = perf_counter()
    signals = compute_page_signals(page, native_blocks, image_boxes)
    signal_seconds = perf_counter() - signal_started_at
    page_kind = classify_page(signals)
    render_dpi = resolve_render_dpi(options)
    background_render_dpi = resolve_background_render_dpi(options)

    page_number = page.number + 1
    has_approved_ocr_blocks = page_number in options.approved_ocr_blocks_by_page
    need_ocr = page_kind in {"scanned", "hybrid"} or not native_blocks
    ocr_blocks: list[TextBlock] = []
    notebooklm_watermark_blocks: list[TextBlock] = []
    page_image: Any | None = None
    ocr_reference_image: Any | None = None
    if has_approved_ocr_blocks:
        ocr_render_started_at = perf_counter()
        page_image = render_page_image(page, dpi=render_dpi)
        ocr_render_seconds = perf_counter() - ocr_render_started_at
        ocr_reference_image = page_image
        ocr_enrich_started_at = perf_counter()
        ocr_blocks = resolve_approved_ocr_blocks(
            page_number=page_number,
            options=options,
            page_rect=page.rect,
            page_image=page_image,
            ocr_engine=ocr_engine,
        )
        ocr_enrich_seconds = perf_counter() - ocr_enrich_started_at
    elif need_ocr:
        if precomputed_ocr_page_data is not None:
            page_image, ocr_page_data = precomputed_ocr_page_data
        else:
            ocr_render_started_at = perf_counter()
            page_image = render_page_image(page, dpi=render_dpi)
            ocr_render_seconds = perf_counter() - ocr_render_started_at

            ocr_started_at = perf_counter()
            ocr_page_data = ocr_engine.extract_text_blocks(page_image, page.number + 1)
            ocr_seconds = perf_counter() - ocr_started_at
        ocr_enrich_started_at = perf_counter()
        ocr_reference_image = ocr_page_data.image
        ocr_blocks = enrich_ocr_blocks(ocr_page_data.blocks, ocr_reference_image)
        ocr_blocks = map_blocks_to_page_coordinates(ocr_blocks, ocr_reference_image.size, page.rect)
        filtered_ocr_blocks = filter_suspicious_ocr_blocks(ocr_blocks, page.rect)
        removed_block_count = len(ocr_blocks) - len(filtered_ocr_blocks)
        if removed_block_count > 0:
            logger.info(
                "Filtered %s suspicious OCR block(s) on page %s",
                removed_block_count,
                page.number + 1,
            )
        ocr_blocks = filtered_ocr_blocks
        ocr_enrich_seconds = perf_counter() - ocr_enrich_started_at

    ocr_blocks, notebooklm_watermark_blocks = split_notebooklm_watermark_blocks(ocr_blocks, page.rect)

    page_mode_started_at = perf_counter()
    text_blocks = select_text_blocks(page_kind, native_blocks, ocr_blocks)

    vector_result = extract_vector_content(
        page,
        text_blocks,
        vector_mode=getattr(options, "vector_mode", "auto"),
        complexity_threshold=int(getattr(options, "vector_complexity_threshold", 80)),
        page_kind=page_kind,
    )
    for table in vector_result.tables:
        apply_cell_fills(table, page)
    text_blocks = filter_text_blocks_outside_tables(text_blocks, vector_result.absorbed_text_block_ids)

    # P2 layout analysis: group lines, bullets, titles, figures; skip headers/footers.
    # Optional Phase-3 VLM planner may replace text grouping when opt-in succeeds.
    if layout_engine is None:
        layout_engine = LayoutEngine(
            engine=str(getattr(options, "layout_engine", "heuristic")),
            model_root=getattr(options, "layout_model_root", None) or options.ocr_model_root,
            skip_header_footer=bool(getattr(options, "layout_skip_header_footer", True)),
        )
    layout_image = ocr_reference_image
    needs_layout_raster = layout_engine.requested_engine == "ppstructure" or layout_planner_service is not None
    if layout_image is None and needs_layout_raster:
        layout_render_started_at = perf_counter()
        layout_image = render_page_image(page, dpi=render_dpi)
        if page_image is None:
            page_image = layout_image
        background_render_seconds += perf_counter() - layout_render_started_at
    text_blocks = split_wide_multiline_blocks(text_blocks, page_width=float(page.rect.width))

    plan_metadata: dict[str, object] = {}
    planner_group_elements: list = []
    used_layout_planner = False
    pending_planner_plan = None
    layout_input_blocks = text_blocks
    if layout_planner_service is not None:
        # Chart-like figure regions from a quick heuristic pass are not available
        # yet; pass native tables as hard obstacles so cells are never merged.
        planned = layout_planner_service.try_plan_page(
            page_image=layout_image,
            text_blocks=text_blocks,
            page_number=page_number,
            page_height=float(page.rect.height),
            page_width=float(page.rect.width),
            table_bboxes=[table.bbox for table in vector_result.tables],
            chart_bboxes=[],
        )
        if planned is not None and planned.used_planner:
            plan_metadata = dict(planned.plan_metadata)
            if planned.frames_applied:
                # Opt-in grouping: planner produced final text frames.
                text_blocks = planned.text_blocks
                planner_group_elements = list(planned.group_elements)
                used_layout_planner = True
            else:
                # Default: keep heuristic paragraph/list grouping; overlay roles later.
                pending_planner_plan = planned.plan

    layout_page = layout_engine.analyze(
        layout_image,
        page_width=float(page.rect.width),
        page_height=float(page.rect.height),
        # Figure/region detection still uses pre-merge geometry when the planner
        # already produced text frames.
        text_blocks=layout_input_blocks if used_layout_planner else text_blocks,
        page_number=page_number,
        shape_bboxes=[shape.bbox for shape in vector_result.shapes],
        table_bboxes=[table.bbox for table in vector_result.tables],
    )
    if used_layout_planner:
        # Keep geometric figure crops; text boxes already come from the planner.
        layout_figure_images = extract_figure_image_placements(
            layout_page.regions,
            page_image=layout_image,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            shape_elements=vector_result.shapes,
            table_elements=vector_result.tables,
        )
        layout_applied_engine = "layout_planner"
        layout_applied_fallback = None
        layout_applied_regions = list(layout_page.regions)
    else:
        layout_applied = apply_layout_to_page(
            text_blocks,
            layout_page,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            page_image=layout_image,
            table_elements=vector_result.tables,
            shape_elements=vector_result.shapes,
            skip_header_footer=bool(getattr(options, "layout_skip_header_footer", True)),
            page_number=page_number,
        )
        text_blocks = layout_applied.text_blocks
        if pending_planner_plan is not None:
            from .layout_planner.apply import annotate_grouped_blocks_with_plan

            text_blocks = annotate_grouped_blocks_with_plan(
                text_blocks,
                pending_planner_plan,
                source_blocks=layout_input_blocks,
                page_height=float(page.rect.height),
                skip_header_footer=bool(getattr(options, "layout_skip_header_footer", True)),
                font_map=getattr(options, "layout_planner_font_map", None),
                apply_fonts=bool(getattr(options, "layout_planner_apply_fonts", False)),
            )
            plan_metadata["frames_applied"] = False
            plan_metadata["heuristic_grouping"] = True
            layout_applied_engine = "layout_planner"
        else:
            layout_applied_engine = layout_applied.engine
        layout_figure_images = list(layout_applied.figure_images)
        layout_applied_fallback = layout_applied.fallback_reason
        layout_applied_regions = layout_applied.regions

    has_remaining_visuals = remaining_visuals_after_nativization(
        image_boxes=image_boxes,
        drawing_count=signals.drawing_count,
        nativized_drawing_count=len(vector_result.nativized_drawing_indices),
        skipped_drawing_count=vector_result.skipped_drawing_count,
    )
    # When every drawing was nativized (or absorbed into tables) and there are no
    # embedded images, treat the page as free of residual raster visuals.
    if vector_result.shapes or vector_result.tables:
        if vector_result.skipped_drawing_count == 0 and not image_boxes:
            if len(vector_result.nativized_drawing_indices) >= signals.drawing_count:
                has_remaining_visuals = False

    quality = score_page(page_kind, text_blocks, native_blocks, ocr_blocks)
    has_native_vectors = bool(vector_result.shapes or vector_result.tables)
    background_mode, fallback_reason = choose_background_mode(
        page_kind=page_kind,
        quality=quality,
        has_text=bool(text_blocks) or has_native_vectors,
        has_visuals=has_remaining_visuals,
        has_native_vectors=has_native_vectors,
    )
    page_mode_seconds = perf_counter() - page_mode_started_at
    logger.debug(
        "Page %s classified as %s with background mode %s (shapes=%s tables=%s)",
        page.number + 1,
        page_kind,
        background_mode,
        len(vector_result.shapes),
        len(vector_result.tables),
    )

    background_image_bytes: bytes | None = None
    image_elements: list[ImagePlacement] = []
    background_inpaint_engine: str | None = None
    background_inpaint_note: str | None = None
    mask_image: Any | None = None
    background_result: BackgroundRenderResult | None = None
    if background_mode == "elements":
        image_elements = extract_image_elements(page, image_boxes, options.render_dpi)
        image_elements = _merge_layout_figures(image_elements, layout_figure_images)
    else:
        effective_background_render_dpi = render_dpi if background_mode == "overlay" else background_render_dpi
        has_ocr_reference_image = ocr_reference_image is not None
        should_downscale_background = effective_background_render_dpi < render_dpi
        uses_original_page_geometry = not options.ocr_use_doc_orientation and not options.use_doc_unwarping
        can_reuse_ocr_raster_for_background = (
            has_ocr_reference_image and should_downscale_background and uses_original_page_geometry
        )
        if page_image is None and effective_background_render_dpi == render_dpi:
            background_render_started_at = perf_counter()
            page_image = render_page_image(page, dpi=render_dpi)
            background_render_seconds += perf_counter() - background_render_started_at
        if ocr_reference_image is not None and effective_background_render_dpi == render_dpi and uses_original_page_geometry:
            background_image = ocr_reference_image
        elif page_image is not None and effective_background_render_dpi == render_dpi:
            background_image = page_image
        elif can_reuse_ocr_raster_for_background:
            # can_reuse_ocr_raster_for_background is only True when
            # has_ocr_reference_image (== ocr_reference_image is not None) is
            # True; make that invariant explicit for the type checker.
            assert ocr_reference_image is not None
            background_render_started_at = perf_counter()
            background_image = resize_rendered_page_image(
                ocr_reference_image,
                source_dpi=render_dpi,
                target_dpi=effective_background_render_dpi,
            )
            background_render_seconds += perf_counter() - background_render_started_at
        else:
            background_render_started_at = perf_counter()
            background_image = render_page_image(page, dpi=effective_background_render_dpi)
            background_render_seconds += perf_counter() - background_render_started_at
        if background_mode == "overlay" and text_blocks:
            mask_blocks = [block for block in text_blocks if block.source == "ocr"] or text_blocks
            mask_blocks = append_notebooklm_watermark_mask_blocks(
                mask_blocks,
                notebooklm_watermark_blocks,
                page_rect=page.rect,
                apply_synthetic_fallback=options.apply_notebooklm_watermark_fallback,
            )
            background_process_started_at = perf_counter()
            background_result = render_overlay_background(
                background_image,
                mask_blocks,
                page.rect,
                options=options,
            )
            background_process_seconds = perf_counter() - background_process_started_at
            background_image = background_result.image
            background_inpaint_engine = background_result.engine_name
            background_inpaint_note = background_result.note
            mask_image = background_result.mask_image
        if has_native_vectors:
            # Peel translucent solid fills off the raster before native shapes
            # re-apply the same alpha (avoids double-darkening caption boxes).
            background_image = uncomposite_translucent_shapes_from_image(
                background_image,
                vector_result.shapes,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
            )
            erase_regions = regions_for_background_erase(
                vector_result.shapes,
                [table.bbox for table in vector_result.tables],
                image_boxes=image_boxes,
            )
            if erase_regions:
                fill_color = resolve_erase_fill_color(vector_result.shapes, background_image)
                background_image = erase_regions_from_image(
                    background_image,
                    erase_regions,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                    fill_color=fill_color,
                    shapes=vector_result.shapes,
                )
                note = background_inpaint_note or ""
                erase_note = f"erased {len(erase_regions)} nativized vector/table region(s)"
                background_inpaint_note = f"{note}; {erase_note}".strip("; ")
            # Opaque fills painted *before* an overlapping picture stay in the
            # raster with that photo; mark them render_under_raster so PPTX
            # paint order matches the PDF (card_93pct, strip-under-photo).
            # Shapes painted after a photo (badge-on-photo, card-after-photo)
            # stay editable above the raster.
            promote_shapes_under_raster(
                vector_result.shapes,
                image_boxes=image_boxes,
                paint_events=paint_events_from_page(page),
            )
        if layout_figure_images or layout_applied_regions:
            # Re-crop figures from the text-erased raster so OCR labels aren't
            # doubled under editable text boxes (scanned_vector_shapes).
            if background_mode == "overlay" and layout_applied_regions:
                layout_figure_images = extract_figure_image_placements(
                    layout_applied_regions,
                    page_image=background_image,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                    shape_elements=vector_result.shapes,
                    table_elements=vector_result.tables,
                )
            if layout_figure_images:
                image_elements = _merge_layout_figures(image_elements, layout_figure_images)
                figure_erase = [placement.bbox for placement in layout_figure_images]
                if figure_erase and background_mode == "overlay":
                    background_image = erase_regions_from_image(
                        background_image,
                        figure_erase,
                        page_width=float(page.rect.width),
                        page_height=float(page.rect.height),
                        fill_color=resolve_erase_fill_color(vector_result.shapes, background_image),
                        shapes=vector_result.shapes,
                    )
        background_encode_started_at = perf_counter()
        background_image_format = options.background_image_format
        if background_mode == "overlay":
            background_image_format = "png"
        background_image_bytes = pil_to_image_bytes(
            background_image,
            image_format=background_image_format,
            jpeg_quality=options.background_jpeg_quality,
        )
        background_encode_seconds = perf_counter() - background_encode_started_at
    if options.debug_dir is not None and ocr_blocks:
        # ocr_blocks is only ever populated inside the has_approved_ocr_blocks
        # or need_ocr branches above, and both of those always set
        # ocr_reference_image too; split_notebooklm_watermark_blocks only
        # partitions the list, so a non-empty ocr_blocks here guarantees
        # ocr_reference_image was set. Make that invariant explicit for the
        # type checker.
        assert ocr_reference_image is not None
        debug_blocks = [block for block in text_blocks if block.source == "ocr"] or ocr_blocks
        debug_masked_image = background_image if background_mode == "overlay" else ocr_reference_image
        if debug_masked_image is not None and ocr_reference_image is not None:
            if debug_masked_image.size != ocr_reference_image.size:
                debug_masked_image = ocr_reference_image
        write_debug_artifacts(
            debug_dir=options.debug_dir,
            page_number=page_number,
            page_image=ocr_reference_image,
            masked_image=debug_masked_image,
            mask_image=mask_image,
            text_blocks=debug_blocks,
            page_rect=page.rect,
            engine_name=background_inpaint_engine,
            engine_note=background_inpaint_note,
            extra_images=(
                background_result.debug_images
                if background_mode == "overlay" and background_result is not None
                else None
            ),
        )
        write_text_fit_debug_report(
            debug_dir=options.debug_dir,
            page_number=page_number,
            text_blocks=debug_blocks,
        )

    page_total_seconds = perf_counter() - page_started_at
    logger.info(
        "Page %s timings: native_extract=%.3fs signals=%.3fs ocr_render=%.3fs ocr=%.3fs ocr_enrich=%.3fs page_mode=%.3fs background_render=%.3fs background_process=%.3fs background_encode=%.3fs total=%.3fs",
        page_number,
        native_extract_seconds,
        signal_seconds,
        ocr_render_seconds,
        ocr_seconds,
        ocr_enrich_seconds,
        page_mode_seconds,
        background_render_seconds,
        background_process_seconds,
        background_encode_seconds,
        page_total_seconds,
    )

    emit_native = background_mode != "full-page"
    return PageResult(
        page_number=page_number,
        page_kind=page_kind,
        background_mode=background_mode,
        width_pt=page.rect.width,
        height_pt=page.rect.height,
        text_blocks=text_blocks if emit_native else [],
        quality_score=quality,
        fallback_reason=fallback_reason,
        background_inpaint_engine=background_inpaint_engine,
        background_inpaint_note=background_inpaint_note,
        background_image_bytes=background_image_bytes,
        image_elements=image_elements,
        shape_elements=vector_result.shapes if emit_native else [],
        table_elements=vector_result.tables if emit_native else [],
        group_elements=planner_group_elements if emit_native else [],
        layout_regions=layout_applied_regions if emit_native else [],
        vector_mode=str(getattr(options, "vector_mode", "auto")),
        vector_fallback_reason=vector_result.fallback_reason,
        layout_engine=layout_applied_engine,
        layout_fallback_reason=layout_applied_fallback,
        plan_metadata=plan_metadata if emit_native else {},
    )


def _merge_layout_figures(
    existing: list[ImagePlacement],
    layout_figures: list[ImagePlacement],
) -> list[ImagePlacement]:
    """Append layout-cropped figures that do not heavily overlap existing images."""

    if not layout_figures:
        return existing
    merged = list(existing)
    for figure in layout_figures:
        overlaps = False
        for prior in merged:
            inter = max(
                0.0,
                min(figure.bbox[2], prior.bbox[2]) - max(figure.bbox[0], prior.bbox[0]),
            ) * max(
                0.0,
                min(figure.bbox[3], prior.bbox[3]) - max(figure.bbox[1], prior.bbox[1]),
            )
            figure_area = max(1.0, (figure.bbox[2] - figure.bbox[0]) * (figure.bbox[3] - figure.bbox[1]))
            if inter / figure_area >= 0.6:
                overlaps = True
                break
        if not overlaps:
            merged.append(figure)
    return merged


def prepare_batched_ocr_page_data(
    document: fitz.Document,
    options: ConversionOptions,
    ocr_engine: OcrEngine,
) -> dict[int, tuple[Image.Image, OcrPageData]]:
    batched_results: dict[int, tuple[Image.Image, OcrPageData]] = {}
    pages_to_ocr: list[tuple[fitz.Page, int]] = []
    for page_index in range(document.page_count):
        page = document[page_index]
        native_blocks, image_boxes = extract_native_text_blocks(page)
        signals = compute_page_signals(page, native_blocks, image_boxes)
        page_kind = classify_page(signals)
        page_number = page.number + 1
        has_approved_ocr_blocks = page_number in options.approved_ocr_blocks_by_page
        need_ocr = page_kind in {"scanned", "hybrid"} or not native_blocks
        if need_ocr and not has_approved_ocr_blocks:
            pages_to_ocr.append((page, page_number))

    render_dpi = resolve_render_dpi(options)
    batch_size = max(1, options.ocr_batch_size or DEFAULT_OCR_BATCH_SIZE)
    for batch_start in range(0, len(pages_to_ocr), batch_size):
        batch = pages_to_ocr[batch_start : batch_start + batch_size]
        batch_images = [render_page_image(page, dpi=render_dpi) for page, _page_number in batch]
        batch_numbers = [page_number for _page, page_number in batch]
        batch_ocr_results = ocr_engine.extract_text_blocks_batch(batch_images, batch_numbers)
        for (page, page_number), page_image, ocr_page_data in zip(batch, batch_images, batch_ocr_results):
            batched_results[page_number] = (page_image, ocr_page_data)
    return batched_results


def resolve_approved_ocr_blocks(
    *,
    page_number: int,
    options: ConversionOptions,
    page_rect: fitz.Rect,
    page_image: Image.Image,
    ocr_engine: OcrEngine,
) -> list[TextBlock]:
    approved_blocks = options.approved_ocr_blocks_by_page.get(page_number, [])
    if not approved_blocks:
        return []

    source_image_size = options.approved_ocr_image_size_by_page.get(page_number, page_image.size)
    scaled_blocks = scale_blocks_to_image_size(approved_blocks, source_image_size, page_image.size)
    resolved_blocks = recognize_missing_approved_block_texts(scaled_blocks, page_image=page_image, ocr_engine=ocr_engine, page_number=page_number)
    resolved_blocks = [block for block in resolved_blocks if block.text.strip()]
    if not resolved_blocks:
        return []
    enriched_blocks = enrich_ocr_blocks(resolved_blocks, page_image)
    return map_blocks_to_page_coordinates(enriched_blocks, page_image.size, page_rect)


def split_notebooklm_watermark_blocks(
    blocks: list[TextBlock],
    page_rect: fitz.Rect,
) -> tuple[list[TextBlock], list[TextBlock]]:
    kept_blocks: list[TextBlock] = []
    watermark_blocks: list[TextBlock] = []
    for block in blocks:
        if is_notebooklm_watermark_block(block, page_rect=page_rect):
            watermark_blocks.append(block)
            continue
        kept_blocks.append(block)
    return kept_blocks, watermark_blocks


def append_notebooklm_watermark_mask_blocks(
    mask_blocks: list[TextBlock],
    detected_watermark_blocks: list[TextBlock],
    *,
    page_rect: fitz.Rect,
    apply_synthetic_fallback: bool = True,
) -> list[TextBlock]:
    if detected_watermark_blocks:
        return [*mask_blocks, *detected_watermark_blocks]
    if not apply_synthetic_fallback:
        return mask_blocks
    return [*mask_blocks, build_notebooklm_watermark_mask_block(page_rect)]


def is_notebooklm_watermark_block(block: TextBlock, *, page_rect: fitz.Rect) -> bool:
    normalized_text = "".join(character.lower() for character in block.text if character.isalnum())
    if NOTEBOOKLM_WATERMARK_TOKEN not in normalized_text:
        return False

    page_width = max(float(page_rect.width), 1.0)
    page_height = max(float(page_rect.height), 1.0)
    x0, y0, x1, y1 = block.bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return (
        x0 >= page_width * NOTEBOOKLM_WATERMARK_RIGHT_MIN_RATIO
        and y0 >= page_height * NOTEBOOKLM_WATERMARK_TOP_MIN_RATIO
        and width <= page_width * NOTEBOOKLM_WATERMARK_MAX_WIDTH_RATIO
        and height <= page_height * NOTEBOOKLM_WATERMARK_MAX_HEIGHT_RATIO
    )


def build_notebooklm_watermark_mask_block(page_rect: fitz.Rect) -> TextBlock:
    page_width = float(page_rect.width)
    page_height = float(page_rect.height)
    right_margin = max(4.0, page_width * NOTEBOOKLM_WATERMARK_FALLBACK_RIGHT_MARGIN_RATIO)
    bottom_margin = max(4.0, page_height * NOTEBOOKLM_WATERMARK_FALLBACK_BOTTOM_MARGIN_RATIO)
    block_width = max(48.0, page_width * NOTEBOOKLM_WATERMARK_FALLBACK_WIDTH_RATIO)
    block_height = max(12.0, page_height * NOTEBOOKLM_WATERMARK_FALLBACK_HEIGHT_RATIO)
    x1 = max(block_width, page_width - right_margin)
    y1 = max(block_height, page_height - bottom_margin)
    x0 = max(0.0, x1 - block_width)
    y0 = max(0.0, y1 - block_height)
    return TextBlock(
        id="synthetic_notebooklm_watermark",
        source="ocr",
        bbox=(x0, y0, x1, y1),
        text="NotebookLM",
        confidence=1.0,
        block_role="watermark",
    )


def recognize_missing_approved_block_texts(
    blocks: list[TextBlock],
    *,
    page_image: Image.Image,
    ocr_engine: OcrEngine,
    page_number: int,
) -> list[TextBlock]:
    resolved_blocks: list[TextBlock] = []
    for index, block in enumerate(blocks, start=1):
        if block.text.strip():
            resolved_blocks.append(block)
            continue
        recognized_text, recognized_confidence = ocr_engine.recognize_text_in_box(
            page_image,
            block.image_bbox or block.bbox,
            page_number=page_number * 1000 + index,
        )
        resolved_blocks.append(
            TextBlock(
                id=block.id,
                source=block.source,
                bbox=block.bbox,
                text=recognized_text,
                confidence=recognized_confidence if recognized_text else block.confidence,
                font_family=block.font_family,
                font_size=block.font_size,
                font_color=block.font_color,
                bold=block.bold,
                italic=block.italic,
                reading_order=block.reading_order,
                block_role=block.block_role,
                image_bbox=block.image_bbox,
                image_polygon=block.image_polygon,
            )
        )
    return resolved_blocks


def scale_blocks_to_image_size(
    blocks: list[TextBlock],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[TextBlock]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0:
        return blocks
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return blocks

    scaled_blocks: list[TextBlock] = []
    for block in blocks:
        x0, y0, x1, y1 = block.bbox
        scaled_blocks.append(
            TextBlock(
                id=block.id,
                source=block.source,
                bbox=(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                text=block.text,
                confidence=block.confidence,
                font_family=block.font_family,
                font_size=block.font_size * scale_y if block.font_size else None,
                font_color=block.font_color,
                bold=block.bold,
                italic=block.italic,
                reading_order=block.reading_order,
                block_role=block.block_role,
                image_bbox=(
                    (
                        block.image_bbox[0] * scale_x,
                        block.image_bbox[1] * scale_y,
                        block.image_bbox[2] * scale_x,
                        block.image_bbox[3] * scale_y,
                    )
                    if block.image_bbox is not None
                    else None
                ),
                image_polygon=(
                    tuple((point[0] * scale_x, point[1] * scale_y) for point in block.image_polygon)
                    if block.image_polygon is not None
                    else None
                ),
            )
        )
    return scaled_blocks


def resize_rendered_page_image(image: Image.Image, *, source_dpi: int, target_dpi: int) -> Image.Image:
    if target_dpi >= source_dpi:
        return image

    scale = target_dpi / max(source_dpi, 1)
    target_width = max(1, int(round(image.width * scale)))
    target_height = max(1, int(round(image.height * scale)))
    return image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)


__all__ = [
    "BackgroundInpaintingEngine",
    "BackgroundInpaintingError",
    "BackgroundRenderResult",
    "ConversionOptions",
    "ConversionReport",
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
    "filter_suspicious_ocr_blocks",
    "fit_text_frame",
    "intersection_ratio",
    "is_cjk",
    "map_blocks_to_page_coordinates",
    "mask_area_ratio",
    "mask_text_regions_with_white_boxes",
    "measure_text_dimensions",
    "most_common_or_none",
    "pil_to_png_bytes",
    "promote_ocr_bold_blocks",
    "pil_to_image_bytes",
    "pt_to_emu",
    "render_overlay_background",
    "render_page_image",
    "render_page_to_slide",
    "resolve_background_inpainting_engine",
    "resolve_background_render_dpi",
    "resolve_ocr_fit_max_size",
    "resolve_vertical_anchor",
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
