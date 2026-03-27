from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw

from .models import TextBlock


def write_debug_artifacts(
    *,
    debug_dir: Path,
    page_number: int,
    page_image: Image.Image,
    masked_image: Image.Image,
    mask_image: Image.Image | None,
    text_blocks: list[TextBlock],
    page_rect: fitz.Rect,
    engine_name: str | None,
    engine_note: str | None,
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
    if mask_image is not None:
        mask_image.convert("L").save(debug_dir / f"{prefix}_mask.png")
    if engine_name or engine_note:
        payload = {
            "page": page_number,
            "engine": engine_name,
            "note": engine_note,
        }
        (debug_dir / f"{prefix}_background.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


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


def clip_to_image(value: float, limit: int) -> int:
    return max(0, min(limit, int(round(value))))
