from __future__ import annotations

import io
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Pt

from .models import ConversionReport, ImagePlacement, PageKind, PageResult, QualityScore, TextBlock


EMU_PER_PT = 12700
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEFAULT_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


@dataclass(slots=True)
class ConversionOptions:
    input_path: Path
    output_path: Path
    report_path: Path
    mode: str = "editable"
    lang: str = "ch"
    render_dpi: int = 144
    debug_dir: Path | None = None
    use_doc_unwarping: bool = False


@dataclass(slots=True)
class PageSignals:
    native_char_count: int
    native_text_area_ratio: float
    image_area_ratio: float
    drawing_count: int


@dataclass(slots=True)
class OcrPageData:
    blocks: list[TextBlock]
    image: Image.Image


class OcrEngine:
    def __init__(self, lang: str, *, use_doc_unwarping: bool) -> None:
        self.lang = lang
        self.use_doc_unwarping = use_doc_unwarping
        self._engine: Any | None = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                lang=self.lang,
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=True,
                use_doc_unwarping=self.use_doc_unwarping,
            )
        return self._engine

    def extract_text_blocks(self, image: Image.Image, page_number: int) -> OcrPageData:
        image_array = np.array(image.convert("RGB"))
        results = self._get_engine().predict(image_array)
        blocks: list[TextBlock] = []
        reference_image = image.convert("RGB")
        order = 1
        for result in results:
            payload = _coerce_ocr_payload(result)
            candidate_image = _extract_ocr_reference_image(payload)
            if candidate_image is not None:
                reference_image = candidate_image
            extracted = _extract_ocr_blocks(payload, page_number=page_number, order_start=order)
            blocks.extend(extracted)
            order += len(extracted)
        return OcrPageData(blocks=blocks, image=reference_image)


def convert_pdf(options: ConversionOptions) -> ConversionReport:
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.report_path.parent.mkdir(parents=True, exist_ok=True)

    ocr_engine = OcrEngine(
        lang=options.lang,
        use_doc_unwarping=options.use_doc_unwarping,
    )
    page_results: list[PageResult] = []

    with fitz.open(options.input_path) as document:
        presentation = Presentation()
        first_page = document[0]
        presentation.slide_width = pt_to_emu(first_page.rect.width)
        presentation.slide_height = pt_to_emu(first_page.rect.height)
        blank_layout = presentation.slide_layouts[6]

        for page_index in range(document.page_count):
            page = document[page_index]
            page_result = analyze_page(page, options, ocr_engine)
            page_results.append(page_result)
            render_page_to_slide(
                presentation,
                blank_layout,
                page_result,
                slide_width_pt=first_page.rect.width,
                slide_height_pt=first_page.rect.height,
            )

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

    background_png: bytes | None = None
    image_elements: list[ImagePlacement] = []
    if background_mode == "elements":
        image_elements = extract_image_elements(page, image_boxes, options.render_dpi)
    else:
        background_image = ocr_reference_image if need_ocr else page_image
        if background_mode == "overlay" and text_blocks:
            mask_blocks = [block for block in text_blocks if block.source == "ocr"] or text_blocks
            background_image = mask_text_regions_with_white_boxes(
                background_image,
                mask_blocks,
                page.rect,
            )
        background_png = pil_to_png_bytes(background_image)

    if options.debug_dir is not None and ocr_blocks:
        write_debug_artifacts(
            debug_dir=options.debug_dir,
            page_number=page.number + 1,
            page_image=ocr_reference_image,
            masked_image=background_image if background_mode == "overlay" else ocr_reference_image,
            text_blocks=[block for block in text_blocks if block.source == "ocr"] or ocr_blocks,
            page_rect=page.rect,
        )
        write_text_fit_debug_report(
            debug_dir=options.debug_dir,
            page_number=page.number + 1,
            text_blocks=[block for block in text_blocks if block.source == "ocr"] or ocr_blocks,
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
        background_png=background_png,
        image_elements=image_elements,
    )


def render_page_to_slide(
    presentation: Presentation,
    blank_layout: Any,
    page_result: PageResult,
    *,
    slide_width_pt: float,
    slide_height_pt: float,
) -> None:
    slide = presentation.slides.add_slide(blank_layout)
    scale_x = slide_width_pt / page_result.width_pt
    scale_y = slide_height_pt / page_result.height_pt

    if page_result.background_png is not None:
        slide.shapes.add_picture(
            io.BytesIO(page_result.background_png),
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
    text_frame.word_wrap = True
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

    font = run.font
    if block.font_family:
        font.name = block.font_family
    if block.font_size:
        font.size = Pt(max(6.0, block.font_size * min(scale_x, scale_y)))
    if block.font_color:
        font.color.rgb = RGBColor.from_string(block.font_color.lstrip("#"))
    font.bold = block.bold
    font.italic = block.italic


def render_page_image(page: fitz.Page, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


def resolve_render_dpi(options: ConversionOptions) -> int:
    if options.mode == "fast":
        return min(options.render_dpi, 110)
    if options.mode == "fidelity":
        return max(options.render_dpi, 180)
    return options.render_dpi


def extract_native_text_blocks(page: fitz.Page) -> tuple[list[TextBlock], list[tuple[float, float, float, float]]]:
    text_dict = page.get_text("dict")
    text_blocks: list[TextBlock] = []
    image_boxes: list[tuple[float, float, float, float]] = []
    order = 1

    for block_index, block in enumerate(text_dict.get("blocks", [])):
        block_type = block.get("type")
        bbox = tuple(float(value) for value in block.get("bbox", (0, 0, 0, 0)))
        if block_type == 1:
            image_boxes.append(bbox)
            continue
        if block_type != 0:
            continue

        lines: list[str] = []
        span_fonts: list[str] = []
        span_sizes: list[float] = []
        span_colors: list[str] = []
        bold = False
        italic = False

        for line in block.get("lines", []):
            parts: list[str] = []
            for span in line.get("spans", []):
                text = str(span.get("text", ""))
                if not text.strip():
                    continue
                parts.append(text)
                font_name = str(span.get("font", "")) or None
                if font_name:
                    span_fonts.append(font_name)
                    lower_font_name = font_name.lower()
                    bold = bold or "bold" in lower_font_name
                    italic = italic or "italic" in lower_font_name or "oblique" in lower_font_name
                span_sizes.append(float(span.get("size", 0.0)))
                color = span.get("color")
                if isinstance(color, int):
                    span_colors.append(int_to_hex_color(color))
            line_text = "".join(parts).strip()
            if line_text:
                lines.append(line_text)

        text = "\n".join(lines).strip()
        if not text:
            continue

        text_blocks.append(
            TextBlock(
                id=f"native_{page.number + 1}_{block_index}",
                source="native",
                bbox=bbox,
                text=text,
                confidence=0.99,
                font_family=most_common_or_none(span_fonts),
                font_size=median(span_sizes) if span_sizes else None,
                font_color=most_common_or_none(span_colors),
                bold=bold,
                italic=italic,
                reading_order=order,
            )
        )
        order += 1

    assign_block_roles(text_blocks)
    return sort_text_blocks(text_blocks), image_boxes


def extract_image_elements(
    page: fitz.Page,
    image_boxes: list[tuple[float, float, float, float]],
    dpi: int,
) -> list[ImagePlacement]:
    image_elements: list[ImagePlacement] = []
    for bbox in image_boxes:
        rect = fitz.Rect(*bbox)
        if rect.width <= 1 or rect.height <= 1:
            continue
        pixmap = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        image_elements.append(ImagePlacement(bbox=bbox, png_bytes=pil_to_png_bytes(image)))
    return image_elements


def compute_page_signals(
    page: fitz.Page,
    native_blocks: list[TextBlock],
    image_boxes: list[tuple[float, float, float, float]],
) -> PageSignals:
    page_area = max(page.rect.width * page.rect.height, 1.0)
    native_char_count = sum(len(re.sub(r"\s+", "", block.text)) for block in native_blocks)
    native_text_area_ratio = sum(bbox_area(block.bbox) for block in native_blocks) / page_area
    image_area_ratio = sum(bbox_area(bbox) for bbox in image_boxes) / page_area
    drawing_count = len(page.get_drawings())
    return PageSignals(
        native_char_count=native_char_count,
        native_text_area_ratio=native_text_area_ratio,
        image_area_ratio=image_area_ratio,
        drawing_count=drawing_count,
    )


def classify_page(signals: PageSignals) -> PageKind:
    if signals.native_char_count >= 40 and signals.native_text_area_ratio >= 0.01:
        if signals.image_area_ratio <= 0.75:
            return "digital"
        return "hybrid"
    if signals.native_char_count <= 12 and signals.image_area_ratio >= 0.45:
        return "scanned"
    if signals.native_char_count <= 12 and signals.drawing_count == 0:
        return "scanned"
    return "hybrid"


def select_text_blocks(
    page_kind: PageKind,
    native_blocks: list[TextBlock],
    ocr_blocks: list[TextBlock],
) -> list[TextBlock]:
    if page_kind == "digital":
        return sort_text_blocks(native_blocks)
    if page_kind == "scanned":
        return sort_text_blocks(ocr_blocks)

    combined = list(native_blocks)
    for ocr_block in ocr_blocks:
        overlap = max(
            (intersection_ratio(ocr_block.bbox, native_block.bbox) for native_block in native_blocks),
            default=0.0,
        )
        if overlap < 0.5:
            combined.append(ocr_block)
    return sort_text_blocks(combined)


def score_page(
    page_kind: PageKind,
    selected_blocks: list[TextBlock],
    native_blocks: list[TextBlock],
    ocr_blocks: list[TextBlock],
) -> QualityScore:
    confidences = [block.confidence for block in selected_blocks]
    text_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    if page_kind == "digital":
        layout_overlap_score = 0.95 if native_blocks else 0.0
        editable_ratio = 1.0 if native_blocks else 0.0
    elif page_kind == "scanned":
        layout_overlap_score = 0.72 if ocr_blocks else 0.0
        editable_ratio = 0.65 if ocr_blocks else 0.0
    else:
        layout_overlap_score = 0.82 if selected_blocks else 0.0
        editable_ratio = 0.85 if native_blocks else (0.55 if ocr_blocks else 0.0)

    style_scores = [1.0 if block.source == "native" else 0.45 for block in selected_blocks]
    style_recovery_score = round(sum(style_scores) / len(style_scores), 4) if style_scores else 0.0

    return QualityScore(
        text_confidence=text_confidence,
        layout_overlap_score=layout_overlap_score,
        style_recovery_score=style_recovery_score,
        editable_ratio=editable_ratio,
    )


def choose_background_mode(
    *,
    page_kind: PageKind,
    quality: QualityScore,
    has_text: bool,
    has_visuals: bool,
) -> tuple[str, str | None]:
    if not has_text or quality.editable_ratio < 0.4:
        return "full-page", "Editable ratio below fallback threshold."
    if page_kind == "digital" and quality.editable_ratio >= 0.8 and not has_visuals:
        return "elements", None
    if page_kind == "digital" and quality.editable_ratio >= 0.8:
        return "overlay", "Retained background to preserve non-text visuals."
    return "overlay", None


def enrich_ocr_blocks(blocks: list[TextBlock], image: Image.Image) -> list[TextBlock]:
    enriched: list[TextBlock] = []
    grayscale = image.convert("L")
    for index, block in enumerate(blocks, start=1):
        crop = safe_crop(image, block.bbox)
        gray_crop = safe_crop(grayscale, block.bbox)
        color = estimate_text_color(crop, gray_crop)
        font_size = estimate_font_size(block.text, block.bbox)
        enriched.append(
            TextBlock(
                id=block.id or f"ocr_{index}",
                source="ocr",
                bbox=block.bbox,
                text=block.text,
                confidence=block.confidence,
                font_family=None,
                font_size=font_size,
                font_color=color,
                bold=False,
                italic=False,
                reading_order=block.reading_order,
                block_role=block.block_role,
                image_bbox=block.image_bbox,
                image_polygon=block.image_polygon,
            )
        )
    assign_block_roles(enriched)
    return sort_text_blocks(enriched)


def map_blocks_to_page_coordinates(
    blocks: list[TextBlock],
    image_size: tuple[int, int],
    page_rect: fitz.Rect,
) -> list[TextBlock]:
    image_width, image_height = image_size
    scale_x = page_rect.width / max(image_width, 1)
    scale_y = page_rect.height / max(image_height, 1)
    mapped: list[TextBlock] = []
    for block in blocks:
        x0, y0, x1, y1 = block.bbox
        mapped.append(
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
                image_bbox=block.image_bbox,
                image_polygon=block.image_polygon,
            )
        )
    return sort_text_blocks(mapped)


def estimate_font_size(text: str, bbox: tuple[float, float, float, float]) -> float:
    height = max(1.0, bbox[3] - bbox[1])
    width = max(1.0, bbox[2] - bbox[0])
    script = classify_text_script(text)
    script_height_ratio = {
        "cjk": 0.74,
        "latin": 0.78,
        "numeric": 0.82,
        "mixed": 0.76,
        "other": 0.76,
    }.get(script, 0.76)
    base_size = max(6.0, min(72.0, height * script_height_ratio))
    font_path = choose_measurement_font(script)
    if font_path is None:
        return base_size

    best_size = base_size
    best_score = float("inf")
    min_size = max(6, int(base_size * 0.65))
    max_size = max(min_size + 2, int(base_size * 1.5) + 2)
    target_width = width * script_target_width_ratio(script, text)
    target_height = height * script_target_height_ratio(script)

    for size in range(min_size, min(max_size, 96) + 1):
        measured_width, measured_height = measure_text_dimensions(text, size, font_path)
        if measured_width <= 0 or measured_height <= 0:
            continue
        width_error = abs(measured_width - target_width) / max(target_width, 1.0)
        height_error = abs(measured_height - target_height) / max(target_height, 1.0)
        overflow_penalty = 0.0
        if measured_width > width * 1.02:
            overflow_penalty += (measured_width - width) / max(width, 1.0)
        if measured_height > height * 1.02:
            overflow_penalty += (measured_height - height) / max(height, 1.0)
        score = width_error * 0.7 + height_error * 1.15 + overflow_penalty * 1.5
        if score < best_score:
            best_score = score
            best_size = float(size)

    return best_size


def classify_text_script(text: str) -> str:
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return "other"
    cjk_count = sum(1 for char in stripped if is_cjk(char))
    latin_count = sum(1 for char in stripped if char.isascii() and char.isalpha())
    numeric_count = sum(1 for char in stripped if char.isdigit())
    counts = {
        "cjk": cjk_count,
        "latin": latin_count,
        "numeric": numeric_count,
    }
    dominant = max(counts, key=counts.get)
    dominant_count = counts[dominant]
    if dominant_count == 0:
        return "other"
    if dominant_count >= len(stripped) * 0.8:
        return dominant
    return "mixed"


def is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def choose_measurement_font(script: str) -> str | None:
    candidates = [DEFAULT_FONT_PATH]
    if script in {"cjk", "mixed"}:
        candidates = [DEFAULT_CJK_FONT_PATH, DEFAULT_FONT_PATH]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def script_target_width_ratio(script: str, text: str) -> float:
    if script == "numeric":
        return 0.88
    if script == "latin":
        return 0.92
    if script == "cjk":
        return 0.94
    if "\n" in text:
        return 0.96
    return 0.93


def script_target_height_ratio(script: str) -> float:
    if script == "numeric":
        return 0.82
    if script == "latin":
        return 0.84
    if script == "cjk":
        return 0.9
    return 0.87


@lru_cache(maxsize=512)
def measure_text_dimensions(text: str, font_size: int, font_path: str) -> tuple[float, float]:
    font = ImageFont.truetype(font_path, font_size)
    lines = text.splitlines() or [text]
    widths: list[float] = []
    heights: list[float] = []
    for line in lines:
        sample = line or " "
        left, top, right, bottom = font.getbbox(sample)
        widths.append(float(right - left))
        heights.append(float(bottom - top))
    line_gap = max(0.0, font_size * 0.15)
    total_height = sum(heights) + max(0, len(lines) - 1) * line_gap
    return max(widths, default=0.0), total_height


def estimate_text_color(color_crop: Image.Image, gray_crop: Image.Image) -> str:
    if color_crop.width == 0 or color_crop.height == 0:
        return "#1F1F1F"
    color_stat = ImageStat.Stat(color_crop)
    gray_stat = ImageStat.Stat(gray_crop)
    mean_luma = gray_stat.mean[0]
    if mean_luma < 120:
        return "#FFFFFF"
    mean_rgb = [int(channel) for channel in color_stat.mean[:3]]
    if max(mean_rgb) - min(mean_rgb) < 10:
        return "#1F1F1F"
    return "#{:02X}{:02X}{:02X}".format(*mean_rgb)


def mask_text_regions_with_white_boxes(
    page_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
) -> Image.Image:
    masked_image = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(masked_image)
    for shape in build_mask_shapes(text_blocks, page_image.size, page_rect):
        if shape["kind"] == "polygon":
            draw.polygon(shape["points"], fill=(255, 255, 255))
        else:
            draw.rectangle(shape["bbox"], fill=(255, 255, 255))

    return masked_image


def write_debug_artifacts(
    *,
    debug_dir: Path,
    page_number: int,
    page_image: Image.Image,
    masked_image: Image.Image,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"page_{page_number:03d}"
    base_image = page_image.convert("RGB")
    shapes = build_mask_shapes(text_blocks, base_image.size, page_rect)

    base_image.save(debug_dir / f"{prefix}_original.png")

    det_overlay = base_image.copy()
    det_draw = ImageDraw.Draw(det_overlay)
    for shape in shapes:
        if shape["kind"] == "polygon":
            det_draw.line(shape["points"] + [shape["points"][0]], fill=(255, 0, 0), width=2)
        else:
            det_draw.rectangle(shape["bbox"], outline=(255, 0, 0), width=2)
    det_overlay.save(debug_dir / f"{prefix}_det_overlay.png")

    masked_base = masked_image.convert("RGBA").copy()
    mask_layer = Image.new("RGBA", masked_base.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)
    outline_draw = ImageDraw.Draw(masked_base)
    for shape in shapes:
        if shape["kind"] == "polygon":
            mask_draw.polygon(shape["points"], fill=(0, 128, 255, 64))
            outline_draw.line(shape["points"] + [shape["points"][0]], fill=(255, 0, 0, 255), width=2)
        else:
            mask_draw.rectangle(shape["bbox"], fill=(0, 128, 255, 64))
            outline_draw.rectangle(shape["bbox"], outline=(255, 0, 0, 255), width=2)
    Image.alpha_composite(masked_base, mask_layer).convert("RGB").save(
        debug_dir / f"{prefix}_mask_overlay.png"
    )
    masked_image.convert("RGB").save(debug_dir / f"{prefix}_masked.png")


def write_text_fit_debug_report(
    *,
    debug_dir: Path,
    page_number: int,
    text_blocks: list[TextBlock],
) -> None:
    entries = [build_text_fit_debug_entry(block) for block in text_blocks if block.source == "ocr"]
    if not entries:
        return

    width_abs_ratios = [abs(entry["width_error_ratio"]) for entry in entries]
    height_abs_ratios = [abs(entry["height_error_ratio"]) for entry in entries]
    payload = {
        "page": page_number,
        "block_count": len(entries),
        "summary": {
            "mean_abs_width_error_ratio": round(sum(width_abs_ratios) / len(width_abs_ratios), 4),
            "mean_abs_height_error_ratio": round(sum(height_abs_ratios) / len(height_abs_ratios), 4),
            "max_abs_width_error_ratio": round(max(width_abs_ratios), 4),
            "max_abs_height_error_ratio": round(max(height_abs_ratios), 4),
        },
        "blocks": entries,
    }
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / f"page_{page_number:03d}_text_fit.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_text_fit_debug_entry(block: TextBlock) -> dict[str, object]:
    script = classify_text_script(block.text)
    font_path = choose_measurement_font(script)
    font_size = max(6.0, block.font_size or 12.0)
    if font_path is None:
        estimated_width = 0.0
        estimated_height = 0.0
    else:
        estimated_width, estimated_height = measure_text_dimensions(
            block.text,
            max(1, int(round(font_size))),
            font_path,
        )

    target_width = max(1.0, block.bbox[2] - block.bbox[0])
    target_height = max(1.0, block.bbox[3] - block.bbox[1])
    width_error = estimated_width - target_width
    height_error = estimated_height - target_height

    return {
        "id": block.id,
        "text": block.text,
        "script": script,
        "font_size_pt": round(font_size, 2),
        "font_path": font_path,
        "target_bbox_pt": {
            "width": round(target_width, 2),
            "height": round(target_height, 2),
        },
        "estimated_ppt_text_pt": {
            "width": round(estimated_width, 2),
            "height": round(estimated_height, 2),
        },
        "width_error_pt": round(width_error, 2),
        "height_error_pt": round(height_error, 2),
        "width_error_ratio": round(width_error / target_width, 4),
        "height_error_ratio": round(height_error / target_height, 4),
    }


def build_mask_shapes(
    text_blocks: list[TextBlock],
    image_size: tuple[int, int],
    page_rect: fitz.Rect,
) -> list[dict[str, object]]:
    image_width, image_height = image_size
    scale_x = image_width / max(page_rect.width, 1.0)
    scale_y = image_height / max(page_rect.height, 1.0)
    shapes: list[dict[str, object]] = []

    for block in text_blocks:
        if block.image_polygon:
            polygon = [
                (clip_to_image(x, image_width), clip_to_image(y, image_height))
                for x, y in block.image_polygon
            ]
            if len(polygon) >= 3:
                shapes.append({"kind": "polygon", "points": polygon})
                continue

        if block.image_bbox is not None:
            x0, y0, x1, y1 = block.image_bbox
            left = max(0, int(np.floor(x0)))
            top = max(0, int(np.floor(y0)))
            right = min(image_width, int(np.ceil(x1)))
            bottom = min(image_height, int(np.ceil(y1)))
        else:
            x0, y0, x1, y1 = block.bbox
            left = max(0, int(np.floor(x0 * scale_x)))
            top = max(0, int(np.floor(y0 * scale_y)))
            right = min(image_width, int(np.ceil(x1 * scale_x)))
            bottom = min(image_height, int(np.ceil(y1 * scale_y)))

        if right <= left or bottom <= top:
            continue
        shapes.append({"kind": "rectangle", "bbox": (left, top, right, bottom)})

    return shapes


def sort_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    ordered = sorted(blocks, key=lambda item: (round(item.bbox[1], 1), round(item.bbox[0], 1)))
    for index, block in enumerate(ordered, start=1):
        block.reading_order = index
    return ordered


def assign_block_roles(blocks: list[TextBlock]) -> None:
    if not blocks:
        return
    sizes = [block.font_size for block in blocks if block.font_size is not None]
    baseline = median(sizes) if sizes else 12.0
    for block in blocks:
        text = block.text.lstrip()
        if block.font_size and block.font_size >= baseline * 1.4:
            block.block_role = "title"
        elif text.startswith(("-", "*", "•")):
            block.block_role = "list"
        else:
            block.block_role = "body"


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


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def intersection_ratio(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    left = max(bbox_a[0], bbox_b[0])
    top = max(bbox_a[1], bbox_b[1])
    right = min(bbox_a[2], bbox_b[2])
    bottom = min(bbox_a[3], bbox_b[3])
    overlap = bbox_area((left, top, right, bottom))
    base = bbox_area(bbox_a)
    if base == 0:
        return 0.0
    return overlap / base


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def safe_crop(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left = max(0, min(width, int(bbox[0])))
    top = max(0, min(height, int(bbox[1])))
    right = max(left, min(width, int(bbox[2])))
    bottom = max(top, min(height, int(bbox[3])))
    return image.crop((left, top, right, bottom))


def int_to_hex_color(value: int) -> str:
    return f"#{value & 0xFFFFFF:06X}"


def most_common_or_none(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _coerce_ocr_payload(result: Any) -> Any:
    if isinstance(result, (dict, list)):
        return result
    if hasattr(result, "res"):
        return result.res
    if hasattr(result, "json"):
        payload = result.json
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            return json.loads(payload)
        return payload
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


def _extract_ocr_reference_image(payload: Any) -> Image.Image | None:
    try:
        if isinstance(payload, dict) and "doc_preprocessor_res" in payload:
            pre = payload["doc_preprocessor_res"]
            output_img = pre["output_img"]
            if isinstance(output_img, np.ndarray):
                return Image.fromarray(output_img).convert("RGB")
    except Exception:
        return None
    return None


def _extract_ocr_blocks(payload: Any, *, page_number: int, order_start: int) -> list[TextBlock]:
    if isinstance(payload, dict):
        if {"rec_texts", "rec_scores", "dt_polys"}.issubset(payload.keys()):
            return _blocks_from_recognized_payload(
                payload["dt_polys"],
                payload["rec_texts"],
                payload["rec_scores"],
                page_number=page_number,
                order_start=order_start,
            )
        for nested_key in ("ocr_result", "result", "res", "data"):
            if nested_key in payload:
                return _extract_ocr_blocks(
                    payload[nested_key], page_number=page_number, order_start=order_start
                )

    if isinstance(payload, list):
        if payload and isinstance(payload[0], list) and len(payload[0]) >= 2:
            blocks: list[TextBlock] = []
            for index, item in enumerate(payload, start=order_start):
                polygon = item[0]
                text_payload = item[1]
                if not isinstance(text_payload, (list, tuple)) or len(text_payload) < 2:
                    continue
                text = str(text_payload[0]).strip()
                if not text:
                    continue
                score = float(text_payload[1])
                blocks.append(
                    TextBlock(
                        id=f"ocr_{page_number}_{index}",
                        source="ocr",
                        bbox=polygon_to_bbox(polygon),
                        text=text,
                        confidence=score,
                        reading_order=index,
                        image_bbox=polygon_to_bbox(polygon),
                        image_polygon=normalize_polygon(polygon),
                    )
                )
            return blocks

        blocks: list[TextBlock] = []
        next_order = order_start
        for item in payload:
            extracted = _extract_ocr_blocks(item, page_number=page_number, order_start=next_order)
            blocks.extend(extracted)
            next_order += len(extracted)
        return blocks

    return []


def _blocks_from_recognized_payload(
    polygons: list[Any],
    texts: list[Any],
    scores: list[Any],
    *,
    page_number: int,
    order_start: int,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for index, (polygon, text, score) in enumerate(zip(polygons, texts, scores), start=order_start):
        normalized_text = str(text).strip()
        if not normalized_text:
            continue
        blocks.append(
            TextBlock(
                id=f"ocr_{page_number}_{index}",
                source="ocr",
                bbox=polygon_to_bbox(polygon),
                text=normalized_text,
                confidence=float(score),
                reading_order=index,
                image_bbox=polygon_to_bbox(polygon),
                image_polygon=normalize_polygon(polygon),
            )
        )
    return blocks


def normalize_polygon(polygon: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in polygon)


def polygon_to_bbox(polygon: Any) -> tuple[float, float, float, float]:
    points = list(polygon)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def clip_to_image(value: float, limit: int) -> int:
    return max(0, min(limit, int(round(value))))
