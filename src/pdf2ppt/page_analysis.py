from __future__ import annotations

import io
import re
from collections import Counter
from statistics import median
from typing import Any

import fitz
from PIL import Image

from .core import PageSignals
from .models import ImagePlacement, PageKind, QualityScore, TextBlock
from .text_style import estimate_font_size, estimate_text_bold, estimate_text_color


def render_page_image(page: fitz.Page, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


def resolve_render_dpi(options: Any) -> int:
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
        bold = estimate_text_bold(block.text, gray_crop)
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
                bold=bold,
                italic=False,
                reading_order=block.reading_order,
                block_role=block.block_role,
                image_bbox=block.image_bbox,
                image_polygon=block.image_polygon,
            )
        )
    assign_block_roles(enriched)
    promote_ocr_bold_blocks(enriched)
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


def promote_ocr_bold_blocks(blocks: list[TextBlock]) -> None:
    for block in blocks:
        if block.bold or block.font_size is None:
            continue
        if block.block_role == "title":
            block.bold = True


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
