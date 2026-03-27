from __future__ import annotations

import io
import logging
from typing import Any

from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE, MSO_VERTICAL_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

from .models import TextBlock
from .text_style import (
    choose_measurement_font,
    classify_text_script,
    default_font_family,
    measure_text_dimensions,
    ocr_fit_height_cap_ratio,
    single_line_fit_width_ratio,
)

logger = logging.getLogger(__name__)

EMU_PER_PT = 12700


def render_page_to_slide(
    presentation: Any,
    blank_layout: Any,
    page_result: Any,
    *,
    slide_width_pt: float,
    slide_height_pt: float,
) -> None:
    slide = presentation.slides.add_slide(blank_layout)
    scale_x = slide_width_pt / page_result.width_pt
    scale_y = slide_height_pt / page_result.height_pt

    if page_result.background_image_bytes is not None:
        slide.shapes.add_picture(
            io.BytesIO(page_result.background_image_bytes),
            0,
            0,
            width=presentation.slide_width,
            height=presentation.slide_height,
        )

    for image_element in page_result.image_elements:
        left, top, width, height = bbox_to_shape_geometry(image_element.bbox, scale_x, scale_y)
        slide.shapes.add_picture(
            io.BytesIO(image_element.png_bytes),
            left,
            top,
            width=width,
            height=height,
        )

    for block in sorted(page_result.text_blocks, key=lambda item: item.reading_order):
        add_text_block(slide, block, scale_x=scale_x, scale_y=scale_y)


def add_text_block(slide: Any, block: TextBlock, *, scale_x: float, scale_y: float) -> None:
    left, top, width, height = bbox_to_shape_geometry(block.bbox, scale_x, scale_y)
    textbox = slide.shapes.add_textbox(left, top, max(width, Emu(1)), max(height, Emu(1)))
    text_frame = textbox.text_frame
    text_frame.word_wrap = should_wrap_text_block(block)
    text_frame.vertical_anchor = resolve_vertical_anchor(block)
    text_frame.margin_left = 0
    text_frame.margin_right = 0
    text_frame.margin_top = 0
    text_frame.margin_bottom = 0
    text_frame.auto_size = MSO_AUTO_SIZE.NONE
    text_frame.clear()

    paragraph = text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.LEFT
    run = paragraph.add_run()
    run.text = block.text
    used_fit_text = fit_text_frame(text_frame, block, scale_x=scale_x, scale_y=scale_y)

    font = run.font
    if block.font_family:
        font.name = block.font_family
    if block.font_size and not used_fit_text:
        font.size = Pt(max(6.0, block.font_size * min(scale_x, scale_y)))
    if block.font_color:
        font.color.rgb = RGBColor.from_string(block.font_color.lstrip("#"))
    font.bold = block.bold
    font.italic = block.italic


def fit_text_frame(text_frame: Any, block: TextBlock, *, scale_x: float, scale_y: float) -> bool:
    if block.source != "ocr" or not block.text.strip():
        return False
    script = classify_text_script(block.text)
    font_path = choose_measurement_font(script)
    if font_path is None:
        logger.debug("No measurement font available for script=%s; skipping fit_text", script)
        return False

    font_family = block.font_family or default_font_family(script)
    base_size = max(6, int(round((block.font_size or 12.0) * min(scale_x, scale_y))))
    max_size = max(
        6,
        resolve_ocr_fit_max_size(
            block,
            font_path=font_path,
            base_size=base_size,
            scale_x=scale_x,
            scale_y=scale_y,
            script=script,
        ),
    )
    try:
        text_frame.fit_text(
            font_family=font_family,
            max_size=max_size,
            bold=block.bold,
            italic=block.italic,
            font_file=font_path,
        )
        return True
    except TypeError:
        logger.debug("python-pptx fit_text rejected OCR block %s", block.id)
        return False


def should_wrap_text_block(block: TextBlock) -> bool:
    if block.source == "ocr" and "\n" not in block.text:
        return False
    return True


def resolve_vertical_anchor(block: TextBlock) -> MSO_VERTICAL_ANCHOR:
    if block.source == "ocr" and "\n" not in block.text:
        return MSO_VERTICAL_ANCHOR.MIDDLE
    return MSO_VERTICAL_ANCHOR.TOP


def resolve_ocr_fit_max_size(
    block: TextBlock,
    *,
    font_path: str,
    base_size: int,
    scale_x: float,
    scale_y: float,
    script: str,
) -> int:
    height_pt = max(1.0, (block.bbox[3] - block.bbox[1]) * scale_y)
    max_size = min(96, max(base_size + 3, int(round(height_pt * ocr_fit_height_cap_ratio(script)))))
    if "\n" in block.text:
        return max_size

    width_pt = max(1.0, (block.bbox[2] - block.bbox[0]) * scale_x)
    width_limit = width_pt * single_line_fit_width_ratio(script)
    size_cap = max(6, max_size)
    for size in range(size_cap, 5, -1):
        measured_width, measured_height = measure_text_dimensions(block.text, size, font_path)
        if measured_width <= width_limit and measured_height <= height_pt * 1.03:
            return size
    return min(base_size, max_size)


def bbox_to_shape_geometry(
    bbox: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
) -> tuple[Emu, Emu, Emu, Emu]:
    x0, y0, x1, y1 = bbox
    left = pt_to_emu(x0 * scale_x)
    top = pt_to_emu(y0 * scale_y)
    width = pt_to_emu(max(1.0, (x1 - x0) * scale_x))
    height = pt_to_emu(max(1.0, (y1 - y0) * scale_y))
    return left, top, width, height


def pt_to_emu(value: float) -> Emu:
    return Emu(int(round(value * EMU_PER_PT)))
