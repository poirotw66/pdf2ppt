"""Erase nativized vector / table regions from raster backgrounds.

When drawings and tables are emitted as native PowerPoint objects they must not
also remain baked into the page background image. For digital pages we fill those
regions with a solid page-background color (sampled from a page-fill shape or
from unmasked corners) so the native objects are not duplicated.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from .models import BBox, ShapeElement


def resolve_erase_fill_color(
    shapes: list[ShapeElement],
    image: Image.Image,
    *,
    default: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int]:
    for shape in shapes:
        if shape.is_page_background and shape.fill_color:
            return _hex_to_rgb(shape.fill_color, default=default)
    return sample_corner_color(image, default=default)


def sample_corner_color(
    image: Image.Image,
    *,
    default: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int]:
    width, height = image.size
    if width < 2 or height < 2:
        return default
    rgb = image.convert("RGB")
    raw_samples = [
        rgb.getpixel((1, 1)),
        rgb.getpixel((width - 2, 1)),
        rgb.getpixel((1, height - 2)),
        rgb.getpixel((width - 2, height - 2)),
    ]
    samples: list[tuple[int, int, int]] = []
    for sample in raw_samples:
        if isinstance(sample, tuple) and len(sample) >= 3:
            samples.append((int(sample[0]), int(sample[1]), int(sample[2])))
        elif isinstance(sample, int):
            samples.append((sample, sample, sample))
    if not samples:
        return default
    # Median-ish via sorting channel-wise midpoints.
    channels = []
    for index in range(3):
        values = sorted(sample[index] for sample in samples)
        channels.append(values[len(values) // 2])
    return (channels[0], channels[1], channels[2])


def erase_regions_from_image(
    image: Image.Image,
    regions: list[BBox],
    *,
    page_width: float,
    page_height: float,
    fill_color: tuple[int, int, int],
    padding_px: int = 1,
) -> Image.Image:
    if not regions:
        return image
    result = image.copy()
    draw = ImageDraw.Draw(result)
    scale_x = result.width / max(page_width, 1.0)
    scale_y = result.height / max(page_height, 1.0)
    for bbox in regions:
        left = max(0, int(bbox[0] * scale_x) - padding_px)
        top = max(0, int(bbox[1] * scale_y) - padding_px)
        right = min(result.width, int(math.ceil(bbox[2] * scale_x)) + padding_px)
        bottom = min(result.height, int(math.ceil(bbox[3] * scale_y)) + padding_px)
        if right <= left or bottom <= top:
            continue
        draw.rectangle((left, top, right, bottom), fill=fill_color)
    return result


def _hex_to_rgb(color: str, *, default: tuple[int, int, int]) -> tuple[int, int, int]:
    text = color.lstrip("#")
    if len(text) != 6:
        return default
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return default


def regions_for_background_erase(
    shapes: list[ShapeElement],
    table_bboxes: list[BBox],
) -> list[BBox]:
    """Regions that must disappear from a raster background.

    Page-background shapes are kept as native shapes; when a raster background
    is still present they should also be erased so they are not doubled.
    """

    regions = [shape.bbox for shape in shapes]
    regions.extend(table_bboxes)
    return regions


__all__ = [
    "erase_regions_from_image",
    "regions_for_background_erase",
    "resolve_erase_fill_color",
    "sample_corner_color",
]
