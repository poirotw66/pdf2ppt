"""Erase nativized vector / table regions from raster backgrounds.

When drawings and tables are emitted as native PowerPoint objects they must not
also remain baked into the page background image. For digital pages we fill those
regions with a local backdrop color (ring-median around the region, falling back
to an underneath opaque fill or a page-sample) so native objects are not
duplicated and white halos do not appear when a large card sits under the raster.

Opaque shapes that sit over embedded images are **not** solid-erased: painting a
flat fill under them would destroy the photo/texture underneath. Translucent
solid fills are instead **un-composited** from the raster
(``bg = (pix - α·fill) / (1 - α)``) so a native translucent shape can sit on top
without double-darkening (see ``photo_texture`` regression).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from .models import BBox, ShapeElement

# Shapes at or above this fill opacity may safely replace their raster underlay.
OPAQUE_FILL_OPACITY = 0.99
# Below this, un-compositing is a no-op / unstable; leave pixels alone.
MIN_UNCOMPOSITE_OPACITY = 0.02
# Overlap ratio vs an image box that means "this shape sits on a photo".
IMAGE_OVERLAP_RATIO = 0.25
# Ring width (pixels) when sampling the local backdrop under an erased region.
ERASE_RING_PX = 3
# Minimum IoU to bind a shape bbox to a ``fill-path`` paint-order event.
PAINT_ORDER_MATCH_IOU = 0.5


PaintKind = Literal["path", "image"]


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


def sample_ring_median_color(
    image: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    ring_px: int = ERASE_RING_PX,
    default: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int] | None:
    """Median RGB of a ring just outside ``[left,top,right,bottom]``.

    Returns ``None`` when the ring is empty (e.g. region covers the whole image).
    """

    if right <= left or bottom <= top or ring_px <= 0:
        return None
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    outer_left = max(0, left - ring_px)
    outer_top = max(0, top - ring_px)
    outer_right = min(width, right + ring_px)
    outer_bottom = min(height, bottom + ring_px)
    if outer_right <= outer_left or outer_bottom <= outer_top:
        return None
    patch = rgb[outer_top:outer_bottom, outer_left:outer_right]
    mask = np.ones(patch.shape[:2], dtype=bool)
    inner_left = max(0, left - outer_left)
    inner_top = max(0, top - outer_top)
    inner_right = min(patch.shape[1], right - outer_left)
    inner_bottom = min(patch.shape[0], bottom - outer_top)
    if inner_right > inner_left and inner_bottom > inner_top:
        mask[inner_top:inner_bottom, inner_left:inner_right] = False
    if not bool(mask.any()):
        return None
    pixels = patch[mask]
    medians = np.median(pixels.astype(np.float64), axis=0)
    return (int(medians[0]), int(medians[1]), int(medians[2]))


def fill_color_underneath_region(
    region: BBox,
    shapes: list[ShapeElement],
) -> tuple[int, int, int] | None:
    """Fill of the smallest opaque shape that fully contains ``region``'s center."""

    cx = (region[0] + region[2]) / 2.0
    cy = (region[1] + region[3]) / 2.0
    region_area = _bbox_area(region)
    candidates: list[tuple[float, ShapeElement]] = []
    for shape in shapes:
        if shape.fill_color is None or shape.fill_opacity < OPAQUE_FILL_OPACITY:
            continue
        left, top, right, bottom = shape.bbox
        if not (left <= cx <= right and top <= cy <= bottom):
            continue
        area = _bbox_area(shape.bbox)
        if area + 1e-6 < region_area:
            continue
        candidates.append((area, shape))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return _hex_to_rgb(candidates[0][1].fill_color or "#FFFFFF", default=(255, 255, 255))


def erase_regions_from_image(
    image: Image.Image,
    regions: list[BBox],
    *,
    page_width: float,
    page_height: float,
    fill_color: tuple[int, int, int],
    padding_px: int = 1,
    shapes: list[ShapeElement] | None = None,
    ring_px: int = ERASE_RING_PX,
) -> Image.Image:
    """Solid-fill erased regions using a local backdrop color when possible.

    Prefer the median of a ``ring_px`` ring around each region (avoids white
    halos when a large card is promoted under the raster). Fall back to the
    opaque fill directly underneath the region, then ``fill_color``.
    """

    if not regions:
        return image
    source = image.convert("RGB")
    result = source.copy()
    draw = ImageDraw.Draw(result)
    scale_x = result.width / max(page_width, 1.0)
    scale_y = result.height / max(page_height, 1.0)
    shape_list = list(shapes or [])

    # Sample every ring on the *pre-erase* raster so earlier fills do not bias later ones.
    planned: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    for bbox in regions:
        left = max(0, int(bbox[0] * scale_x) - padding_px)
        top = max(0, int(bbox[1] * scale_y) - padding_px)
        right = min(result.width, int(math.ceil(bbox[2] * scale_x)) + padding_px)
        bottom = min(result.height, int(math.ceil(bbox[3] * scale_y)) + padding_px)
        if right <= left or bottom <= top:
            continue
        color = sample_ring_median_color(
            source,
            left,
            top,
            right,
            bottom,
            ring_px=ring_px,
            default=fill_color,
        )
        if color is None:
            color = fill_color_underneath_region(bbox, shape_list)
        if color is None:
            color = fill_color
        planned.append((left, top, right, bottom, color))

    for left, top, right, bottom, color in planned:
        draw.rectangle((left, top, right, bottom), fill=color)
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


def _bbox_iou(a: BBox, b: BBox) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _shape_overlaps_image(shape_bbox: BBox, image_boxes: list[BBox]) -> bool:
    """True when the shape covers a meaningful fraction of a picture (or vice versa).

    Large page fills must be judged against *picture* area: intersection /
    shape_area is tiny for a full-page rect over a small photo, which used to
    let erase paint over the photo. Small shapes on photos still match via
    intersection / shape_area.
    """

    shape_area = _bbox_area(shape_bbox)
    for image_bbox in image_boxes:
        overlap = _intersection_area(shape_bbox, image_bbox)
        if overlap <= 0:
            continue
        image_area = _bbox_area(image_bbox)
        if image_area > 0 and overlap / image_area >= IMAGE_OVERLAP_RATIO:
            return True
        if shape_area > 0 and overlap / shape_area >= IMAGE_OVERLAP_RATIO:
            return True
    return False


def parse_bboxlog_paint_events(
    bboxlog: list[tuple[Any, Any]] | None,
) -> list[tuple[int, PaintKind, BBox]]:
    """Normalize ``page.get_bboxlog()`` into ``(order, kind, bbox)`` events."""

    events: list[tuple[int, PaintKind, BBox]] = []
    if not bboxlog:
        return events
    for index, entry in enumerate(bboxlog):
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            continue
        kind_raw, bbox_raw = entry[0], entry[1]
        kind_text = str(kind_raw).lower()
        try:
            bbox = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        except (TypeError, ValueError, IndexError):
            continue
        if "image" in kind_text:
            events.append((index, "image", bbox))
        elif "path" in kind_text:
            events.append((index, "path", bbox))
    return events


def paint_events_from_page(page: Any) -> list[tuple[int, PaintKind, BBox]] | None:
    """Read PDF paint order from ``page.get_bboxlog()`` when available.

    Returns ``None`` when ``get_bboxlog`` raises so callers can fall back to the
    conservative size/overlap promotion rule instead of treating a hard failure
    as an empty paint log (which would promote nothing).
    """

    try:
        bboxlog = page.get_bboxlog()
    except Exception:  # noqa: BLE001 - PyMuPDF may raise on odd pages
        return None
    return parse_bboxlog_paint_events(list(bboxlog or []))


def _match_paint_order(
    bbox: BBox,
    events: list[tuple[int, PaintKind, BBox]],
    *,
    kind: PaintKind,
) -> int | None:
    best_index: int | None = None
    best_iou = 0.0
    for order, event_kind, event_bbox in events:
        if event_kind != kind:
            continue
        iou = _bbox_iou(bbox, event_bbox)
        if iou > best_iou:
            best_iou = iou
            best_index = order
    if best_index is None or best_iou < PAINT_ORDER_MATCH_IOU:
        return None
    return best_index


def shape_should_erase_background(
    shape: ShapeElement,
    *,
    image_boxes: list[BBox] | None = None,
) -> bool:
    """Whether a nativized shape's raster underlay may be solid-filled away."""

    # Page fills are drawn beneath the raster; erasing them destroys pictures.
    if shape.is_page_background:
        return False
    # Stroke-only / no-fill shapes have nothing to un-composite as a solid fill.
    if shape.fill_color is None:
        return False
    if shape.fill_opacity < OPAQUE_FILL_OPACITY:
        return False
    if image_boxes and _shape_overlaps_image(shape.bbox, image_boxes):
        return False
    return True


def shape_should_render_under_raster(
    shape: ShapeElement,
    *,
    image_boxes: list[BBox] | None = None,
    paint_events: list[tuple[int, PaintKind, BBox]] | None = None,
) -> bool:
    """Whether an opaque fill must sit under the page raster in PPTX z-order.

    Promote when the fill overlaps an embedded picture enough that erase skipped
    it, and either:

    * PDF paint order is known and drew the fill **before** that picture
      (size is ignored — small strips under larger photos still promote), or
    * paint order is unknown (``paint_events is None``) and the fill is not
      smaller than that picture (conservative badge-safe fallback).

    True page-background fills are handled via ``is_page_background`` instead.
    """

    if shape.is_page_background:
        return False
    if shape.render_under_raster:
        return True
    if shape.fill_color is None:
        return False
    if shape.fill_opacity < OPAQUE_FILL_OPACITY:
        return False
    if not image_boxes:
        return False

    shape_area = _bbox_area(shape.bbox)
    shape_order = None
    if paint_events:
        shape_order = _match_paint_order(shape.bbox, paint_events, kind="path")

    for image_bbox in image_boxes:
        overlap = _intersection_area(shape.bbox, image_bbox)
        if overlap <= 0:
            continue
        image_area = _bbox_area(image_bbox)
        covers_picture = image_area > 0 and overlap / image_area >= IMAGE_OVERLAP_RATIO
        sits_on_picture = shape_area > 0 and overlap / shape_area >= IMAGE_OVERLAP_RATIO
        if not (covers_picture or sits_on_picture):
            continue
        if paint_events is None:
            # Without paint order, keep the conservative PR #12 size rule so
            # badges / icons smaller than the picture stay editable on top.
            if shape_area + 1e-6 < image_area:
                continue
            return True
        if shape_order is None:
            continue
        image_order = _match_paint_order(image_bbox, paint_events, kind="image")
        if image_order is None:
            continue
        if shape_order < image_order:
            return True
    return False


# Backwards-compatible alias used by earlier PR #12 helpers/tests.
shape_should_render_under_background = shape_should_render_under_raster


def promote_shapes_under_raster(
    shapes: list[ShapeElement],
    *,
    image_boxes: list[BBox] | None = None,
    paint_events: list[tuple[int, PaintKind, BBox]] | None = None,
) -> int:
    """Set ``render_under_raster`` on fills that must sit under the page picture.

    Does **not** mutate ``is_page_background`` (reserved for true ≥95% page fills).
    Returns how many shapes were newly marked.
    """

    promoted = 0
    for shape in shapes:
        if shape.is_page_background or shape.render_under_raster:
            continue
        if shape_should_render_under_raster(
            shape,
            image_boxes=image_boxes,
            paint_events=paint_events,
        ):
            shape.render_under_raster = True
            promoted += 1
    return promoted


# Backwards-compatible alias.
promote_shapes_under_background = promote_shapes_under_raster


def regions_for_background_erase(
    shapes: list[ShapeElement],
    table_bboxes: list[BBox],
    *,
    image_boxes: list[BBox] | None = None,
) -> list[BBox]:
    """Regions that must disappear from a raster background.

    Only opaque filled shapes that do not sit on / cover embedded images are
    erased. Page-background fills are skipped. Tables are always erased (their
    grid is fully replaced by a native table).
    """

    regions = [
        shape.bbox
        for shape in shapes
        if shape_should_erase_background(shape, image_boxes=image_boxes)
    ]
    regions.extend(table_bboxes)
    return regions


__all__ = [
    "ERASE_RING_PX",
    "OPAQUE_FILL_OPACITY",
    "erase_regions_from_image",
    "fill_color_underneath_region",
    "paint_events_from_page",
    "parse_bboxlog_paint_events",
    "promote_shapes_under_background",
    "promote_shapes_under_raster",
    "regions_for_background_erase",
    "resolve_erase_fill_color",
    "sample_corner_color",
    "sample_ring_median_color",
    "shape_should_erase_background",
    "shape_should_render_under_background",
    "shape_should_render_under_raster",
    "shape_should_uncomposite_background",
    "uncomposite_translucent_shapes_from_image",
]
