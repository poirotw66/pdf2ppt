from __future__ import annotations

import io
from collections import Counter
from statistics import median

import fitz
from PIL import Image

from .models import ImagePlacement, TextBlock


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


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def int_to_hex_color(value: int) -> str:
    return f"#{value & 0xFFFFFF:06X}"


def most_common_or_none(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]
