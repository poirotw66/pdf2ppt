from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass, field
from statistics import median

import fitz
from PIL import Image

from .models import BBox, ImagePlacement, TextBlock


@dataclass(slots=True)
class _NativeLinePayload:
    text: str
    bbox: BBox
    fonts: list[str] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    bold: bool = False
    italic: bool = False


def extract_native_text_blocks(page: fitz.Page) -> tuple[list[TextBlock], list[tuple[float, float, float, float]]]:
    text_dict = page.get_text("dict")
    text_blocks: list[TextBlock] = []
    image_boxes: list[tuple[float, float, float, float]] = []
    order = 1
    page_width = float(page.rect.width) or 1.0

    for block_index, block in enumerate(text_dict.get("blocks", [])):
        block_type = block.get("type")
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = (float(value) for value in block.get("bbox", (0, 0, 0, 0)))
        bbox = (bbox_x0, bbox_y0, bbox_x1, bbox_y1)
        if block_type == 1:
            image_boxes.append(bbox)
            continue
        if block_type != 0:
            continue

        line_payloads: list[_NativeLinePayload] = []
        for line in block.get("lines", []):
            parts: list[str] = []
            span_fonts: list[str] = []
            span_sizes: list[float] = []
            span_colors: list[str] = []
            bold = False
            italic = False
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
            if not line_text:
                continue
            line_bbox_raw = line.get("bbox", block.get("bbox", (0, 0, 0, 0)))
            line_bbox = (
                float(line_bbox_raw[0]),
                float(line_bbox_raw[1]),
                float(line_bbox_raw[2]),
                float(line_bbox_raw[3]),
            )
            line_payloads.append(
                _NativeLinePayload(
                    text=line_text,
                    bbox=line_bbox,
                    fonts=span_fonts,
                    sizes=span_sizes,
                    colors=span_colors,
                    bold=bold,
                    italic=italic,
                )
            )

        if not line_payloads:
            continue

        # Split horizontally disjoint same-band lines (flowchart labels) into
        # separate text boxes; keep vertically stacked lines as one paragraph.
        for group_index, group in enumerate(_group_native_line_payloads(line_payloads, page_width=page_width)):
            texts = [item.text for item in group]
            boxes = [item.bbox for item in group]
            fonts: list[str] = []
            sizes: list[float] = []
            colors: list[str] = []
            bold = False
            italic = False
            for item in group:
                fonts.extend(item.fonts)
                sizes.extend(item.sizes)
                colors.extend(item.colors)
                bold = bold or item.bold
                italic = italic or item.italic
            group_bbox = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            text_blocks.append(
                TextBlock(
                    id=f"native_{page.number + 1}_{block_index}_{group_index}",
                    source="native",
                    bbox=group_bbox,
                    text="\n".join(texts).strip(),
                    confidence=0.99,
                    font_family=most_common_or_none(fonts),
                    font_size=median(sizes) if sizes else None,
                    font_color=most_common_or_none(colors),
                    bold=bold,
                    italic=italic,
                    reading_order=order,
                )
            )
            order += 1

    assign_block_roles(text_blocks)
    return sort_text_blocks(text_blocks), image_boxes


def _group_native_line_payloads(
    line_payloads: list[_NativeLinePayload],
    *,
    page_width: float,
) -> list[list[_NativeLinePayload]]:
    """Group stacked paragraph lines; keep wide horizontal gaps as separate boxes."""

    if not line_payloads:
        return []
    groups: list[list[_NativeLinePayload]] = [[line_payloads[0]]]
    for payload in line_payloads[1:]:
        previous = groups[-1][-1]
        prev_bbox = previous.bbox
        cand_bbox = payload.bbox
        prev_height = max(1.0, prev_bbox[3] - prev_bbox[1])
        cand_height = max(1.0, cand_bbox[3] - cand_bbox[1])
        vertical_overlap = max(
            0.0,
            min(prev_bbox[3], cand_bbox[3]) - max(prev_bbox[1], cand_bbox[1]),
        )
        min_height = min(prev_height, cand_height)
        same_band = min_height > 0 and vertical_overlap / min_height >= 0.45
        horizontal_gap = max(
            0.0,
            max(cand_bbox[0] - prev_bbox[2], prev_bbox[0] - cand_bbox[2]),
        )
        if same_band and horizontal_gap > page_width * 0.08:
            groups.append([payload])
            continue
        # Vertically stacked continuation of the same column/paragraph.
        vertical_gap = cand_bbox[1] - prev_bbox[3]
        avg_height = (prev_height + cand_height) / 2.0
        if (not same_band) and 0 <= vertical_gap <= avg_height * 1.35:
            groups[-1].append(payload)
            continue
        if same_band and horizontal_gap <= avg_height * 0.6:
            groups[-1].append(payload)
            continue
        groups.append([payload])
    return groups


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
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
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
    return pil_to_image_bytes(image, image_format="png")


def pil_to_image_bytes(
    image: Image.Image,
    *,
    image_format: str,
    jpeg_quality: int = 82,
) -> bytes:
    buffer = io.BytesIO()
    normalized_format = image_format.strip().lower()
    if normalized_format == "jpeg":
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=max(1, min(95, int(jpeg_quality))),
            optimize=True,
        )
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def int_to_hex_color(value: int) -> str:
    return f"#{value & 0xFFFFFF:06X}"


def most_common_or_none(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]
