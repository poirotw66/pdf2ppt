"""Erase nativized vector / table regions from raster backgrounds.

When drawings and tables are emitted as native PowerPoint objects they must not
also remain baked into the page background image. For digital pages we fill those
regions with a solid page-background color (sampled from a page-fill shape or
from unmasked corners) so the native objects are not duplicated.

Opaque shapes that sit over embedded images are **not** solid-erased: painting a
flat fill under them would destroy the photo/texture underneath. Translucent
solid fills are instead **un-composited** from the raster
(``bg = (pix - α·fill) / (1 - α)``) so a native translucent shape can sit on top
without double-darkening (see ``photo_texture`` regression).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .models import BBox, ShapeElement

# Shapes at or above this fill opacity may safely replace their raster underlay.
OPAQUE_FILL_OPACITY = 0.99
# Below this, un-compositing is a no-op / unstable; leave pixels alone.
MIN_UNCOMPOSITE_OPACITY = 0.02
# Overlap ratio vs an image box that means "this shape sits on a photo".
IMAGE_OVERLAP_RATIO = 0.25


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


def shape_should_uncomposite_background(shape: ShapeElement) -> bool:
    """Whether a nativized translucent solid fill must be peeled off the raster."""

    if shape.fill_color is None:
        return False
    if shape.fill_opacity >= OPAQUE_FILL_OPACITY:
        return False
    if shape.fill_opacity < MIN_UNCOMPOSITE_OPACITY:
        return False
    return True


def uncomposite_translucent_shapes_from_image(
    image: Image.Image,
    shapes: list[ShapeElement],
    *,
    page_width: float,
    page_height: float,
    padding_px: int = 0,
) -> Image.Image:
    """Reverse ``out = (1-α)·bg + α·fill`` inside each translucent shape bbox.

    Native PowerPoint will re-apply the same translucent fill, so the raster must
    hold the pre-tint background. Skipping this step double-darkens caption boxes
    over photos (photo_texture).
    """

    targets = [shape for shape in shapes if shape_should_uncomposite_background(shape)]
    if not targets:
        return image

    result = image.convert("RGB").copy()
    pixels = np.asarray(result, dtype=np.float32)
    scale_x = result.width / max(page_width, 1.0)
    scale_y = result.height / max(page_height, 1.0)

    for shape in targets:
        alpha = float(shape.fill_opacity)
        fill_rgb = _hex_to_rgb(shape.fill_color or "#000000", default=(0, 0, 0))
        left = max(0, int(shape.bbox[0] * scale_x) - padding_px)
        top = max(0, int(shape.bbox[1] * scale_y) - padding_px)
        right = min(result.width, int(math.ceil(shape.bbox[2] * scale_x)) + padding_px)
        bottom = min(result.height, int(math.ceil(shape.bbox[3] * scale_y)) + padding_px)
        if right <= left or bottom <= top:
            continue
        region = pixels[top:bottom, left:right]
        fill = np.asarray(fill_rgb, dtype=np.float32)
        restored = (region - alpha * fill) / max(1e-6, 1.0 - alpha)
        pixels[top:bottom, left:right] = np.clip(restored, 0.0, 255.0)

    return Image.fromarray(np.round(pixels).astype(np.uint8), mode="RGB")


def _hex_to_rgb(color: str, *, default: tuple[int, int, int]) -> tuple[int, int, int]:
    text = color.lstrip("#")
    if len(text) != 6:
        return default
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return default


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def _shape_overlaps_image(shape_bbox: BBox, image_boxes: list[BBox]) -> bool:
    area = _bbox_area(shape_bbox)
    if area <= 0:
        return False
    for image_bbox in image_boxes:
        if _intersection_area(shape_bbox, image_bbox) / area >= IMAGE_OVERLAP_RATIO:
            return True
    return False


def shape_should_erase_background(
    shape: ShapeElement,
    *,
    image_boxes: list[BBox] | None = None,
) -> bool:
    """Whether a nativized shape's raster underlay may be solid-filled away."""

    if shape.fill_opacity < OPAQUE_FILL_OPACITY:
        return False
    if image_boxes and _shape_overlaps_image(shape.bbox, image_boxes):
        return False
    return True


def regions_for_background_erase(
    shapes: list[ShapeElement],
    table_bboxes: list[BBox],
    *,
    image_boxes: list[BBox] | None = None,
) -> list[BBox]:
    """Regions that must disappear from a raster background.

    Only opaque shapes that do not sit on embedded images are erased. Tables are
    always erased (their grid is fully replaced by a native table).
    """

    regions = [
        shape.bbox
        for shape in shapes
        if shape_should_erase_background(shape, image_boxes=image_boxes)
    ]
    regions.extend(table_bboxes)
    return regions


__all__ = [
    "OPAQUE_FILL_OPACITY",
    "erase_regions_from_image",
    "regions_for_background_erase",
    "resolve_erase_fill_color",
    "sample_corner_color",
    "shape_should_erase_background",
    "shape_should_uncomposite_background",
    "uncomposite_translucent_shapes_from_image",
]
