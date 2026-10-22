"""Deterministic vector-shape and table nativization for digital PDF pages (P1).

Converts PyMuPDF ``page.get_drawings()`` paths into ``ShapeElement`` instances and
``page.find_tables()`` results into ``TableElement`` instances suitable for
native PowerPoint objects. Complex paths (gradients, clipping, huge path counts)
fall back to the raster background path.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any

import fitz
import numpy as np

from .models import (
    ArrowDirection,
    BBox,
    ShapeElement,
    ShapeKind,
    TableCell,
    TableElement,
    TextBlock,
)

logger = logging.getLogger(__name__)

# Full-page (or near-full-page) fills at or above this fraction become page-background
# shapes. Opaque fills covering ~95%+ of the page with no stroke are treated as the
# slide backdrop, never as foreground content that can cover pictures.
PAGE_BACKGROUND_AREA_RATIO = 0.95
# Paths with more than this many items are treated as complex (raster fallback).
DEFAULT_MAX_PATH_ITEMS_AUTO = 24
DEFAULT_MAX_PATH_ITEMS_AGGRESSIVE = 64
# Max cubic segments allowed before preferring raster (ellipses use 4).
DEFAULT_MAX_CURVES_AUTO = 8
DEFAULT_MAX_CURVES_AGGRESSIVE = 24
# Soft cap on emitted shapes; beyond this, dense pages fall back.
DEFAULT_MAX_SHAPES_AUTO = 120
DEFAULT_MAX_SHAPES_AGGRESSIVE = 400
# Overlap ratio for associating a text block with a table cell.
TABLE_TEXT_OVERLAP_RATIO = 0.45
# Drawing rect mostly inside a table → absorb into table (skip as shape).
TABLE_DRAWING_CONTAINMENT_RATIO = 0.85
# find_tables() false positive: tiny cell-count candidates covering almost the
# whole page (e.g. two side-by-side page fills misread as a 1×2 table).
SPURIOUS_TABLE_MAX_CELLS = 2
SPURIOUS_TABLE_PAGE_RATIO = 0.80
# Inset when matching a cell to a containing filled vector rect.
CELL_FILL_VECTOR_INSET_PT = 2.0
# Modal non-text pixel share required to accept a raster-sampled cell fill.
CELL_FILL_MODAL_MIN_SHARE = 0.60
# Opaque enough to trust the vector fill colour (else use rendered pixels).
CELL_FILL_OPAQUE_OPACITY = 0.99
# Word/Excel hairline gridlines drawn as thin filled rects, not strokes.
HAIRLINE_RULE_MAX_SHORT_PT = 1.5
HAIRLINE_RULE_MIN_LONG_PT = 8.0
# Quantise RGB to this many bits/channel before the modal share test.
CELL_FILL_QUANTIZE_BITS = 4
# Backwards-compatible alias (pre-tighten empty-cell guard threshold).
LARGE_EMPTY_CELL_PAGE_RATIO = SPURIOUS_TABLE_PAGE_RATIO


@dataclass(slots=True)
class VectorExtractionResult:
    shapes: list[ShapeElement]
    tables: list[TableElement]
    # Drawing indices that were nativized (for background masking).
    nativized_drawing_indices: list[int]
    # BBoxes to remove from the raster background (shapes + tables).
    mask_regions: list[BBox]
    # Text-block ids absorbed into native tables (must not also become text boxes).
    absorbed_text_block_ids: set[str]
    fallback_reason: str | None = None
    skipped_drawing_count: int = 0


def resolve_complexity_limits(vector_mode: str, threshold: int) -> tuple[int, int, int, int]:
    """Return (max_drawings, max_path_items, max_curves, max_shapes)."""

    mode = (vector_mode or "auto").strip().lower()
    base = max(1, int(threshold))
    if mode == "aggressive":
        return (
            max(base, 400),
            DEFAULT_MAX_PATH_ITEMS_AGGRESSIVE,
            DEFAULT_MAX_CURVES_AGGRESSIVE,
            DEFAULT_MAX_SHAPES_AGGRESSIVE,
        )
    return (
        base,
        DEFAULT_MAX_PATH_ITEMS_AUTO,
        DEFAULT_MAX_CURVES_AUTO,
        DEFAULT_MAX_SHAPES_AUTO,
    )


def should_attempt_vector_nativization(
    *,
    vector_mode: str,
    page_kind: str,
    drawing_count: int,
    complexity_threshold: int,
) -> tuple[bool, str | None]:
    mode = (vector_mode or "auto").strip().lower()
    if mode == "off":
        return False, "vector_mode=off"
    if page_kind == "scanned":
        return False, "scanned_page"
    max_drawings, _, _, _ = resolve_complexity_limits(mode, complexity_threshold)
    if drawing_count > max_drawings:
        return False, f"drawing_count={drawing_count} exceeds threshold={max_drawings}"
    return True, None


def extract_vector_content(
    page: fitz.Page,
    text_blocks: list[TextBlock],
    *,
    vector_mode: str = "auto",
    complexity_threshold: int = 80,
    page_kind: str = "digital",
) -> VectorExtractionResult:
    """Extract native shapes and tables for one page.

    Scanned pages and ``vector_mode=off`` return empty results. Complex drawings
    are skipped individually; if the page is too dense overall, shape extraction
    is skipped while tables may still succeed.
    """

    empty = VectorExtractionResult(
        shapes=[],
        tables=[],
        nativized_drawing_indices=[],
        mask_regions=[],
        absorbed_text_block_ids=set(),
    )
    mode = (vector_mode or "auto").strip().lower()
    if mode == "off":
        empty.fallback_reason = "vector_mode=off"
        return empty
    if page_kind == "scanned":
        empty.fallback_reason = "scanned_page"
        return empty

    drawings = list(page.get_drawings())
    attempt, reason = should_attempt_vector_nativization(
        vector_mode=mode,
        page_kind=page_kind,
        drawing_count=len(drawings),
        complexity_threshold=complexity_threshold,
    )
    max_drawings, max_path_items, max_curves, max_shapes = resolve_complexity_limits(
        mode, complexity_threshold
    )

    tables = extract_tables(page, text_blocks, page_number=page.number + 1)
    table_bboxes = [table.bbox for table in tables]
    borderless_table_bboxes = [
        table.bbox
        for table in tables
        if (table.border_width or 0) <= 0 and (table.inner_border_width or 0) <= 0
    ]
    absorbed_ids: set[str] = set()
    for table in tables:
        for cell in table.cells:
            absorbed_ids.update(cell.text_block_ids)
    # Also absorb any text block that mostly sits inside a table bbox. Native PDF
    # text often arrives as multi-cell column/row blocks that only partially
    # overlap individual cells, so cell-level matching alone is insufficient.
    for block in text_blocks:
        if block.id in absorbed_ids:
            continue
        for table_bbox in table_bboxes:
            if intersection_ratio(block.bbox, table_bbox) >= 0.5:
                absorbed_ids.add(block.id)
                break
            cx = (block.bbox[0] + block.bbox[2]) / 2.0
            cy = (block.bbox[1] + block.bbox[3]) / 2.0
            if table_bbox[0] <= cx <= table_bbox[2] and table_bbox[1] <= cy <= table_bbox[3]:
                absorbed_ids.add(block.id)
                break

    shapes: list[ShapeElement] = []
    nativized_indices: list[int] = []
    skipped = 0
    page_area = max(float(page.rect.width * page.rect.height), 1.0)

    if not attempt:
        # Still emit tables when the page is dense; shapes stay raster.
        mask_regions = [table.bbox for table in tables]
        return VectorExtractionResult(
            shapes=[],
            tables=tables,
            nativized_drawing_indices=[],
            mask_regions=mask_regions,
            absorbed_text_block_ids=absorbed_ids,
            fallback_reason=reason,
            skipped_drawing_count=len(drawings),
        )

    for index, path in enumerate(drawings):
        if _drawing_belongs_to_table(
            path,
            table_bboxes,
            page_area=page_area,
            borderless_table_bboxes=borderless_table_bboxes,
        ):
            nativized_indices.append(index)
            continue
        shape = _path_to_shape(
            path,
            page_number=page.number + 1,
            drawing_index=index,
            page_area=page_area,
            max_path_items=max_path_items,
            max_curves=max_curves,
        )
        if shape is None:
            skipped += 1
            continue
        shapes.extend(split_page_background_stroke(shape))
        nativized_indices.append(index)
        if len(shapes) > max_shapes:
            # Too dense — abandon shapes, keep tables.
            logger.info(
                "Page %s: shape count exceeded %s; falling back shapes to raster",
                page.number + 1,
                max_shapes,
            )
            return VectorExtractionResult(
                shapes=[],
                tables=tables,
                nativized_drawing_indices=[],
                mask_regions=[table.bbox for table in tables],
                absorbed_text_block_ids=absorbed_ids,
                fallback_reason=f"shape_count_exceeded_{max_shapes}",
                skipped_drawing_count=len(drawings),
            )

    # Hairline row/column rules kept beside borderless inferred tables must paint
    # above the table (white body fills otherwise hide them in spTree order).
    _raise_hairlines_above_borderless_tables(shapes, tables)

    mask_regions = [shape.bbox for shape in shapes if not shape.is_page_background]
    mask_regions.extend(table.bbox for table in tables)

    return VectorExtractionResult(
        shapes=shapes,
        tables=tables,
        nativized_drawing_indices=nativized_indices,
        mask_regions=mask_regions,
        absorbed_text_block_ids=absorbed_ids,
        fallback_reason=None if shapes or tables else reason,
        skipped_drawing_count=skipped,
    )


def _raise_hairlines_above_borderless_tables(
    shapes: list[ShapeElement],
    tables: list[TableElement],
) -> None:
    borderless = [
        table
        for table in tables
        if (table.border_width or 0) <= 0 and (table.inner_border_width or 0) <= 0
    ]
    if not borderless:
        return
    max_table_z = max((int(table.z_order) for table in borderless), default=0)
    bump = 1
    for shape in shapes:
        height = abs(shape.bbox[3] - shape.bbox[1])
        width = abs(shape.bbox[2] - shape.bbox[0])
        if min(width, height) > HAIRLINE_RULE_MAX_SHORT_PT + 0.5:
            continue
        if max(width, height) < HAIRLINE_RULE_MIN_LONG_PT:
            continue
        if not any(_bbox_center_inside(shape.bbox, table.bbox, padding=2.0) for table in borderless):
            continue
        shape.render_above_tables = True
        shape.z_order = max_table_z + bump
        bump += 1


def extract_tables(
    page: fitz.Page,
    text_blocks: list[TextBlock],
    *,
    page_number: int,
) -> list[TableElement]:
    """Detect ruled tables plus borderless / zebra grids from aligned text.

    Ruled grids come from PyMuPDF ``find_tables()``. Borderless column-aligned
    text and alternating row-fill (zebra) tables are recovered by
    :mod:`pdf2ppt.table_detection` when they do not overlap a ruled hit.
    """

    finder = None
    for kwargs in ({"refine": True}, {}):
        try:
            finder = page.find_tables(**kwargs)
            break
        except TypeError:
            continue
        except Exception as error:  # noqa: BLE001 - PyMuPDF table finder can raise broadly
            logger.debug("find_tables(%s) failed on page %s: %s", kwargs, page_number, error)
            finder = None
            break

    span_index = _collect_page_spans(page)
    tables: list[TableElement] = []
    if finder is not None:
        for table_index, table in enumerate(getattr(finder, "tables", []) or []):
            element = _table_to_element(
                table,
                text_blocks=text_blocks,
                page_number=page_number,
                table_index=table_index,
                span_index=span_index,
                page=page,
            )
            if element is not None:
                tables.append(element)

    # Borderless / zebra fallback (lazy import avoids import cycle).
    from .table_detection import detect_inferred_tables

    inferred = detect_inferred_tables(
        page,
        text_blocks,
        page_number=page_number,
        existing_table_bboxes=[table.bbox for table in tables],
    )
    tables.extend(inferred)
    return tables


def filter_text_blocks_outside_tables(
    text_blocks: list[TextBlock],
    absorbed_ids: set[str],
) -> list[TextBlock]:
    if not absorbed_ids:
        return text_blocks
    return [block for block in text_blocks if block.id not in absorbed_ids]


def remaining_visuals_after_nativization(
    *,
    image_boxes: list[BBox],
    drawing_count: int,
    nativized_drawing_count: int,
    skipped_drawing_count: int,
) -> bool:
    """Whether a raster overlay is still required after vector nativization."""

    if image_boxes:
        return True
    remaining_drawings = drawing_count - nativized_drawing_count
    return remaining_drawings > 0 or skipped_drawing_count > 0


def rgb_tuple_to_hex(color: Any) -> str | None:
    if color is None:
        return None
    if isinstance(color, (int, float)):
        # Gray
        value = int(round(max(0.0, min(1.0, float(color))) * 255))
        return f"#{value:02X}{value:02X}{value:02X}"
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        channels = [int(round(max(0.0, min(1.0, float(channel))) * 255)) for channel in color[:3]]
        return f"#{channels[0]:02X}{channels[1]:02X}{channels[2]:02X}"
    return None


def parse_dash_style(dashes: Any) -> str | None:
    if dashes is None:
        return None
    text = str(dashes).strip()
    if not text or text in {"[] 0", "[]0"}:
        return "solid"
    return "dash"


def bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def is_page_background_fill(
    bbox: BBox,
    *,
    page_area: float,
    fill_color: str | None,
    stroke_color: str | None = None,
    stroke_width: float | None = None,
    area_ratio: float = PAGE_BACKGROUND_AREA_RATIO,
) -> bool:
    """Whether a filled path should be treated as a page/slide background.

    Any fill covering ``area_ratio`` of the page is a backdrop, including stroked
    frames. Callers should split a visible stroke onto a separate outline shape
    via :func:`split_page_background_stroke` so the border can sit above content
    without the opaque fill covering pictures or titles.
    """

    _ = stroke_color, stroke_width  # stroke no longer disqualifies a page fill
    if fill_color is None or page_area <= 0:
        return False
    return bbox_area(bbox) / page_area >= area_ratio


def split_page_background_stroke(shape: ShapeElement) -> list[ShapeElement]:
    """Split a stroked page-background fill into fill (bottom) + outline (top).

    The fill stays ``is_page_background``; the outline is a no-fill foreground
    shape so the border remains visible above a full-slide raster.
    """

    if not shape.is_page_background:
        return [shape]
    has_visible_stroke = shape.stroke_color is not None and (
        shape.stroke_width is None or float(shape.stroke_width) > 0.0
    )
    if not has_visible_stroke:
        return [shape]

    fill_only = ShapeElement(
        id=shape.id,
        kind=shape.kind,
        bbox=shape.bbox,
        points=list(shape.points),
        fill_color=shape.fill_color,
        stroke_color=None,
        stroke_width=None,
        stroke_dash=None,
        fill_opacity=shape.fill_opacity,
        stroke_opacity=1.0,
        z_order=shape.z_order,
        provenance=shape.provenance,
        confidence=shape.confidence,
        locked=shape.locked,
        drawing_index=shape.drawing_index,
        is_page_background=True,
        arrow_direction=shape.arrow_direction,
    )
    outline = ShapeElement(
        id=f"{shape.id}_outline",
        kind=shape.kind,
        bbox=shape.bbox,
        points=list(shape.points),
        fill_color=None,
        stroke_color=shape.stroke_color,
        stroke_width=shape.stroke_width,
        stroke_dash=shape.stroke_dash,
        fill_opacity=1.0,
        stroke_opacity=shape.stroke_opacity,
        # Above the page fill / typical early drawings; text is raised later.
        z_order=max(shape.z_order + 1, 1),
        provenance=shape.provenance,
        confidence=shape.confidence,
        locked=shape.locked,
        drawing_index=shape.drawing_index,
        is_page_background=False,
        arrow_direction=shape.arrow_direction,
    )
    return [fill_only, outline]


def intersection_area(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def intersection_ratio(inner: BBox, outer: BBox) -> float:
    area = bbox_area(inner)
    if area <= 0:
        return 0.0
    return intersection_area(inner, outer) / area


def _rect_to_bbox(rect: Any) -> BBox:
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def _point_xy(point: Any) -> tuple[float, float]:
    return (float(point.x), float(point.y))


def _drawing_belongs_to_table(
    path: dict[str, Any],
    table_bboxes: list[BBox],
    *,
    page_area: float,
    borderless_table_bboxes: list[BBox] | None = None,
) -> bool:
    if not table_bboxes:
        return False
    rect = path.get("rect")
    if rect is None:
        return False
    bbox = _rect_to_bbox(rect)
    # Never absorb a page-sized fill into a table.
    if bbox_area(bbox) / page_area >= PAGE_BACKGROUND_AREA_RATIO:
        return False
    hairline = _is_hairline_rule_path(path, bbox)
    for table_bbox in table_bboxes:
        inside = intersection_ratio(bbox, table_bbox) >= TABLE_DRAWING_CONTAINMENT_RATIO or _bbox_center_inside(
            bbox, table_bbox, padding=1.0
        )
        if not inside:
            continue
        # Borderless inferred tables clear PPTX borders; keep hairline row/column
        # separators as shapes so they are not dropped (b_header_band_fillrules).
        if hairline and borderless_table_bboxes:
            if any(
                intersection_ratio(table_bbox, other) >= 0.9 or intersection_ratio(other, table_bbox) >= 0.9
                for other in borderless_table_bboxes
            ):
                continue
        return True
    return False


def _is_hairline_rule_path(path: dict[str, Any], bbox: BBox) -> bool:
    """True for stroked thin rules or Word/Excel hairline filled-rect separators."""

    width = abs(bbox[2] - bbox[0])
    height = abs(bbox[3] - bbox[1])
    short_side = min(width, height)
    long_side = max(width, height)
    path_type = path.get("type")
    if path_type in {"s", "fs"}:
        stroke_width = path.get("width")
        if path.get("color") is None or stroke_width is None:
            return False
        return 0.0 < float(stroke_width) <= 2.0 and long_side >= HAIRLINE_RULE_MIN_LONG_PT
    if path.get("fill") is not None and short_side <= HAIRLINE_RULE_MAX_SHORT_PT:
        return long_side >= HAIRLINE_RULE_MIN_LONG_PT
    return False


def _bbox_center_inside(inner: BBox, outer: BBox, *, padding: float = 0.0) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return (
        outer[0] - padding <= cx <= outer[2] + padding
        and outer[1] - padding <= cy <= outer[3] + padding
    )


def _path_to_shape(
    path: dict[str, Any],
    *,
    page_number: int,
    drawing_index: int,
    page_area: float,
    max_path_items: int,
    max_curves: int,
) -> ShapeElement | None:
    path_type = path.get("type")
    if path_type in {"clip", "group"}:
        return None

    items = list(path.get("items") or [])
    if not items:
        return None
    if len(items) > max_path_items:
        return None

    curve_count = sum(1 for item in items if item and item[0] == "c")
    if curve_count > max_curves:
        return None

    fill = rgb_tuple_to_hex(path.get("fill"))
    stroke = rgb_tuple_to_hex(path.get("color"))
    width = path.get("width")
    stroke_width = float(width) if width is not None else None
    raw_fill_opacity = path.get("fill_opacity")
    raw_stroke_opacity = path.get("stroke_opacity")
    fill_opacity = float(raw_fill_opacity) if isinstance(raw_fill_opacity, (int, float)) else 1.0
    stroke_opacity = float(raw_stroke_opacity) if isinstance(raw_stroke_opacity, (int, float)) else 1.0
    dash = parse_dash_style(path.get("dashes"))
    rect = path.get("rect")
    if rect is None:
        return None
    bbox = _rect_to_bbox(rect)
    if bbox_area(bbox) <= 0 and len(items) == 1 and items[0][0] != "l":
        return None

    is_page_background = is_page_background_fill(
        bbox,
        page_area=page_area,
        fill_color=fill,
        stroke_color=stroke,
        stroke_width=stroke_width,
    )

    kind, points = _classify_path_items(items, bbox=bbox, close_path=bool(path.get("closePath")))
    if kind is None:
        return None

    # Degenerate zero-area lines still need endpoints.
    if kind == "line" and len(points) >= 2:
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    arrow_direction: ArrowDirection | None = None
    if kind == "arrow":
        arrow_direction = _infer_arrow_direction(points, bbox)

    return ShapeElement(
        id=f"shape_{page_number}_{drawing_index}",
        kind=kind,
        bbox=bbox,
        points=points,
        fill_color=fill,
        stroke_color=stroke,
        stroke_width=stroke_width,
        stroke_dash=dash,
        fill_opacity=max(0.0, min(1.0, fill_opacity)),
        stroke_opacity=max(0.0, min(1.0, stroke_opacity)),
        z_order=int(path.get("seqno") or drawing_index),
        drawing_index=drawing_index,
        is_page_background=is_page_background,
        confidence=0.95 if kind in {"rect", "line", "ellipse"} else 0.75,
        arrow_direction=arrow_direction,
    )


def _classify_path_items(
    items: list[tuple[Any, ...]],
    *,
    bbox: BBox,
    close_path: bool,
) -> tuple[ShapeKind | None, list[tuple[float, float]]]:
    if len(items) == 1 and items[0][0] == "re":
        return "rect", []

    if len(items) == 1 and items[0][0] == "l":
        return "line", [_point_xy(items[0][1]), _point_xy(items[0][2])]

    if len(items) == 1 and items[0][0] == "qu":
        quad = items[0][1]
        points = [_point_xy(quad.ul), _point_xy(quad.ur), _point_xy(quad.lr), _point_xy(quad.ll)]
        return "polygon", points

    if _looks_like_ellipse(items, bbox):
        return "ellipse", []

    if _looks_like_rounded_rect(items, bbox):
        return "round_rect", []

    if _looks_like_arrow(items, bbox):
        return "arrow", _polyline_points(items)  # direction set on ShapeElement below

    only_lines = all(item[0] == "l" for item in items)
    if only_lines:
        points = _polyline_points(items)
        if len(points) < 2:
            return None, []
        if close_path or _points_form_closed_loop(points):
            if len(points) >= 3:
                return "polygon", points
        return "polyline" if len(points) > 2 else "line", points

    # Approximate freeform from line/curve endpoints (skip control points for beziers).
    points = _polyline_points(items, include_curve_controls=False)
    if len(points) >= 3:
        return "freeform", points
    if len(points) == 2:
        return "line", points
    return None, []


def _polyline_points(
    items: list[tuple[Any, ...]],
    *,
    include_curve_controls: bool = False,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def append_point(point: tuple[float, float]) -> None:
        if points and math.hypot(points[-1][0] - point[0], points[-1][1] - point[1]) < 1e-6:
            return
        points.append(point)

    for item in items:
        kind = item[0]
        if kind == "l":
            append_point(_point_xy(item[1]))
            append_point(_point_xy(item[2]))
        elif kind == "c":
            append_point(_point_xy(item[1]))
            if include_curve_controls:
                append_point(_point_xy(item[2]))
                append_point(_point_xy(item[3]))
            append_point(_point_xy(item[4]))
        elif kind == "re":
            rect = item[1]
            append_point((float(rect.x0), float(rect.y0)))
            append_point((float(rect.x1), float(rect.y0)))
            append_point((float(rect.x1), float(rect.y1)))
            append_point((float(rect.x0), float(rect.y1)))
        elif kind == "qu":
            quad = item[1]
            for corner in (quad.ul, quad.ur, quad.lr, quad.ll):
                append_point(_point_xy(corner))
    return points


def _points_form_closed_loop(points: list[tuple[float, float]], *, tol: float = 1.5) -> bool:
    if len(points) < 3:
        return False
    return math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= tol


def _looks_like_ellipse(items: list[tuple[Any, ...]], bbox: BBox) -> bool:
    if len(items) != 4 or any(item[0] != "c" for item in items):
        return False
    width = max(1e-6, bbox[2] - bbox[0])
    height = max(1e-6, bbox[3] - bbox[1])
    # Four cubic arcs whose endpoints sit near mid-sides of the bbox.
    endpoints = [_point_xy(item[1]) for item in items] + [_point_xy(items[-1][4])]
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    side_hits = 0
    for x, y in endpoints:
        on_left = abs(x - bbox[0]) < width * 0.08 and abs(y - cy) < height * 0.2
        on_right = abs(x - bbox[2]) < width * 0.08 and abs(y - cy) < height * 0.2
        on_top = abs(y - bbox[1]) < height * 0.08 and abs(x - cx) < width * 0.2
        on_bottom = abs(y - bbox[3]) < height * 0.08 and abs(x - cx) < width * 0.2
        if on_left or on_right or on_top or on_bottom:
            side_hits += 1
    return side_hits >= 4


def _looks_like_rounded_rect(items: list[tuple[Any, ...]], bbox: BBox) -> bool:
    if len(items) < 6:
        return False
    line_count = sum(1 for item in items if item[0] == "l")
    curve_count = sum(1 for item in items if item[0] == "c")
    if curve_count < 4 or line_count < 2:
        return False
    width = max(1e-6, bbox[2] - bbox[0])
    height = max(1e-6, bbox[3] - bbox[1])
    # Aspect and item mix typical of rounded rectangles drawn as lines+quarter arcs.
    return 0.15 <= width / height <= 8.0 and curve_count <= 8


def _looks_like_arrow(items: list[tuple[Any, ...]], bbox: BBox) -> bool:
    points = _polyline_points(items)
    if len(points) < 5 or len(points) > 12:
        return False
    width = max(1e-6, bbox[2] - bbox[0])
    height = max(1e-6, bbox[3] - bbox[1])
    # Elongated closed polygon often used for simple arrow heads / chevrons.
    if not _points_form_closed_loop(points) and not (items and items[0][0] == "l"):
        return False
    return (width / height >= 1.8) or (height / width >= 1.8)


def _infer_arrow_direction(points: list[tuple[float, float]], bbox: BBox) -> ArrowDirection:
    """Pick the MSO arrow facing the sharp tip of a closed arrow polygon.

    Along the long axis, the tip is the end with fewer vertices (typically a
    single extreme point). Counting by distance-from-centroid fails when a
    fat tail corner is farther from centre than the tip (vector_shapes.pdf).
    """

    unique = _unique_points(points)
    if len(unique) < 3:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return "right" if width >= height else "down"

    width = max(1e-6, bbox[2] - bbox[0])
    height = max(1e-6, bbox[3] - bbox[1])
    horizontal = width >= height
    # Band thickness for "at this end" — a few percent of the long axis.
    band = max(2.0, (width if horizontal else height) * 0.08)

    if horizontal:
        left_count = sum(1 for x, _y in unique if abs(x - bbox[0]) <= band)
        right_count = sum(1 for x, _y in unique if abs(x - bbox[2]) <= band)
        if right_count < left_count:
            return "right"
        if left_count < right_count:
            return "left"
        # Tie-break: end whose extreme vertex is closer to mid-height (sharper tip).
        mid_y = (bbox[1] + bbox[3]) / 2.0
        left_pts = [p for p in unique if abs(p[0] - bbox[0]) <= band]
        right_pts = [p for p in unique if abs(p[0] - bbox[2]) <= band]
        left_score = min(abs(p[1] - mid_y) for p in left_pts) if left_pts else 1e9
        right_score = min(abs(p[1] - mid_y) for p in right_pts) if right_pts else 1e9
        return "right" if right_score <= left_score else "left"

    top_count = sum(1 for _x, y in unique if abs(y - bbox[1]) <= band)
    bottom_count = sum(1 for _x, y in unique if abs(y - bbox[3]) <= band)
    if bottom_count < top_count:
        return "down"
    if top_count < bottom_count:
        return "up"
    mid_x = (bbox[0] + bbox[2]) / 2.0
    top_pts = [p for p in unique if abs(p[1] - bbox[1]) <= band]
    bottom_pts = [p for p in unique if abs(p[1] - bbox[3]) <= band]
    top_score = min(abs(p[0] - mid_x) for p in top_pts) if top_pts else 1e9
    bottom_score = min(abs(p[0] - mid_x) for p in bottom_pts) if bottom_pts else 1e9
    return "down" if bottom_score <= top_score else "up"


def _unique_points(
    points: list[tuple[float, float]],
    *,
    tol: float = 1e-3,
) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for point in points:
        if any(math.hypot(point[0] - other[0], point[1] - other[1]) <= tol for other in unique):
            continue
        unique.append(point)
    return unique


def _table_to_element(
    table: Any,
    *,
    text_blocks: list[TextBlock],
    page_number: int,
    table_index: int,
    span_index: list[dict[str, Any]],
    page: fitz.Page,
) -> TableElement | None:
    row_count = int(getattr(table, "row_count", 0) or 0)
    col_count = int(getattr(table, "col_count", 0) or 0)
    if row_count < 1 or col_count < 1:
        return None

    bbox = _rect_to_bbox(fitz.Rect(table.bbox))
    rows = list(getattr(table, "rows", []) or [])
    extract = table.extract() if hasattr(table, "extract") else None
    placements = getattr(table, "placements", None)

    cells: list[TableCell] = []
    covered: set[tuple[int, int]] = set()

    if placements:
        for row_index, placement_row in enumerate(placements):
            col_index = 0
            for span in placement_row or []:
                while (row_index, col_index) in covered:
                    col_index += 1
                if col_index >= col_count:
                    break
                colspan = max(1, int(getattr(span, "colspan", 1) or 1))
                rowspan = max(1, int(getattr(span, "rowspan", 1) or 1))
                text = str(getattr(span, "text", "") or "")
                cell_bbox = _span_bbox(span, rows, row_index, col_index, colspan, rowspan, bbox)
                cell = _build_table_cell(
                    row=row_index,
                    col=col_index,
                    text=text,
                    rowspan=rowspan,
                    colspan=colspan,
                    cell_bbox=cell_bbox,
                    text_blocks=text_blocks,
                    span_index=span_index,
                )
                cells.append(cell)
                for r in range(row_index, row_index + rowspan):
                    for c in range(col_index, col_index + colspan):
                        if r != row_index or c != col_index:
                            covered.add((r, c))
                col_index += colspan
    else:
        for row_index, row in enumerate(rows):
            row_cells = list(getattr(row, "cells", []) or [])
            while len(row_cells) < col_count:
                row_cells.append(None)
            col_index = 0
            while col_index < col_count:
                if (row_index, col_index) in covered:
                    col_index += 1
                    continue
                cell_rect = row_cells[col_index]
                if cell_rect is None:
                    col_index += 1
                    continue
                # Subsequent None cells in the same row are covered by this merge.
                colspan = 1
                look = col_index + 1
                while look < col_count and row_cells[look] is None:
                    colspan += 1
                    look += 1
                text = ""
                if extract is not None and row_index < len(extract) and col_index < len(extract[row_index]):
                    raw = extract[row_index][col_index]
                    text = "" if raw is None else str(raw)
                cell_bbox = _rect_to_bbox(fitz.Rect(cell_rect))
                if colspan > 1:
                    row_bbox = _rect_to_bbox(fitz.Rect(row.bbox))
                    col_width = (row_bbox[2] - row_bbox[0]) / max(col_count, 1)
                    cell_bbox = (
                        row_bbox[0] + col_index * col_width,
                        cell_bbox[1],
                        row_bbox[0] + (col_index + colspan) * col_width,
                        cell_bbox[3],
                    )
                cell = _build_table_cell(
                    row=row_index,
                    col=col_index,
                    text=text,
                    rowspan=1,
                    colspan=colspan,
                    cell_bbox=cell_bbox,
                    text_blocks=text_blocks,
                    span_index=span_index,
                )
                cells.append(cell)
                for c in range(col_index + 1, col_index + colspan):
                    covered.add((row_index, c))
                col_index += colspan

    if not cells:
        return None

    column_widths, row_heights = _estimate_table_geometry(rows, bbox, row_count, col_count)
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    if _is_spurious_table_candidate(
        cells,
        table_bbox=bbox,
        page_area=page_area,
        page=page,
    ):
        logger.debug(
            "Rejecting table %s on page %s: spurious fill-edge / page-covering candidate",
            table_index,
            page_number,
        )
        return None

    border_color, border_width, outer_width, inner_width = _infer_table_border_style(page, bbox)
    return TableElement(
        id=f"table_{page_number}_{table_index}",
        bbox=bbox,
        row_count=row_count,
        col_count=col_count,
        cells=cells,
        column_widths=column_widths,
        row_heights=row_heights,
        border_color=border_color or "#4D4D4D",
        border_width=border_width if border_width is not None else 1.0,
        outer_border_width=outer_width,
        inner_border_width=inner_width,
        z_order=10_000 + table_index,
        confidence=0.9,
    )


def _is_spurious_table_candidate(
    cells: list[TableCell],
    *,
    table_bbox: BBox,
    page_area: float,
    page: fitz.Page,
    max_cells: int = SPURIOUS_TABLE_MAX_CELLS,
    page_ratio: float = SPURIOUS_TABLE_PAGE_RATIO,
) -> bool:
    """True for find_tables() false positives (page fills, not ruled tables).

    Reject when:

    * the candidate has ``<= max_cells`` cells and covers ``>= page_ratio`` of the
      page (side-by-side page fills misread as a 1×2 table), or
    * the grid is derived only from fill edges (no stroked rules near the bbox)
      **and** there is no alternating zebra row-fill stack to justify it.
    """

    if page_area <= 0 or not cells:
        return False
    covers_page = bbox_area(table_bbox) / page_area >= page_ratio
    if len(cells) <= max_cells and covers_page:
        return True
    if _table_has_stroked_rules(page, table_bbox):
        return False
    if _table_has_zebra_row_fills(page, table_bbox):
        return False
    return True


def _table_has_zebra_row_fills(page: fitz.Page, table_bbox: BBox) -> bool:
    """True when stacked wide fills under ``table_bbox`` alternate colours."""

    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return False
    bands: list[tuple[BBox, str]] = []
    for path in drawings:
        fill = path.get("fill")
        if fill is None:
            continue
        fill_hex = rgb_tuple_to_hex(fill)
        if fill_hex is None:
            continue
        rect = path.get("rect")
        if rect is None:
            continue
        line_bbox = _rect_to_bbox(rect)
        if intersection_ratio(line_bbox, table_bbox) < 0.5:
            continue
        width = abs(line_bbox[2] - line_bbox[0])
        height = abs(line_bbox[3] - line_bbox[1])
        if height < 10.0 or height > 64.0 or width < height * 2.5:
            continue
        bands.append((line_bbox, fill_hex))
    if len(bands) < 3:
        return False
    bands.sort(key=lambda item: item[0][1])
    colors = [color for _, color in bands]
    if len(set(colors)) < 2:
        return False
    alternations = sum(1 for left, right in zip(colors, colors[1:]) if left != right)
    return alternations >= max(1, len(bands) - 2)


def _table_has_stroked_rules(page: fitz.Page, table_bbox: BBox, *, pad: float = 2.0) -> bool:
    """True when the table has stroked rules or Word/Excel hairline filled-rect rules.

    Large page fills (``probe_straddle``) are not rules: only thin filled rects with
    short side ≤1.5pt and long side ≥8pt count as hairline gridlines.
    """

    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return False
    outer = (
        table_bbox[0] - pad,
        table_bbox[1] - pad,
        table_bbox[2] + pad,
        table_bbox[3] + pad,
    )
    for path in drawings:
        rect = path.get("rect")
        if rect is None:
            continue
        line_bbox = _rect_to_bbox(rect)
        if not _bbox_center_inside(line_bbox, outer, padding=pad):
            continue
        path_type = path.get("type")
        if path_type in {"s", "fs"}:
            color = path.get("color")
            width = path.get("width")
            if color is None or width is None:
                continue
            if float(width) <= 0 or float(width) > 4.0:
                continue
            return True
        # Hairline filled rects (Word/Excel gridlines exported as fills, not strokes).
        if path_type == "f" and path.get("fill") is not None:
            width = abs(line_bbox[2] - line_bbox[0])
            height = abs(line_bbox[3] - line_bbox[1])
            short_side = min(width, height)
            long_side = max(width, height)
            if short_side <= HAIRLINE_RULE_MAX_SHORT_PT and long_side >= HAIRLINE_RULE_MIN_LONG_PT:
                return True
    return False


def _span_bbox(
    span: Any,
    rows: list[Any],
    row_index: int,
    col_index: int,
    colspan: int,
    rowspan: int,
    table_bbox: BBox,
) -> BBox:
    span_bbox = getattr(span, "bbox", None)
    if span_bbox is not None:
        return _rect_to_bbox(fitz.Rect(span_bbox))
    if row_index < len(rows):
        row = rows[row_index]
        row_cells = list(getattr(row, "cells", []) or [])
        if col_index < len(row_cells) and row_cells[col_index] is not None:
            return _rect_to_bbox(fitz.Rect(row_cells[col_index]))
    return table_bbox


def _build_table_cell(
    *,
    row: int,
    col: int,
    text: str,
    rowspan: int,
    colspan: int,
    cell_bbox: BBox,
    text_blocks: list[TextBlock],
    span_index: list[dict[str, Any]],
) -> TableCell:
    overlapping = [
        block
        for block in text_blocks
        if intersection_ratio(block.bbox, cell_bbox) >= TABLE_TEXT_OVERLAP_RATIO
        or intersection_ratio(cell_bbox, block.bbox) >= TABLE_TEXT_OVERLAP_RATIO
    ]
    if not text.strip() and overlapping:
        text = "\n".join(block.text for block in overlapping if block.text.strip())

    style = _style_from_spans(cell_bbox, span_index)
    if style["font_size"] is None:
        font_sizes = [block.font_size for block in overlapping if block.font_size]
        style["font_size"] = float(median(font_sizes)) if font_sizes else None
    if style["font_color"] is None:
        font_colors = [block.font_color for block in overlapping if block.font_color]
        style["font_color"] = Counter(font_colors).most_common(1)[0][0] if font_colors else "#1A1A1A"
    if not style["bold"]:
        style["bold"] = any(block.bold for block in overlapping)

    return TableCell(
        row=row,
        col=col,
        text=text.strip(),
        rowspan=rowspan,
        colspan=colspan,
        fill_color=None,
        font_size=style["font_size"],
        font_color=style["font_color"],
        font_name=style.get("font_name"),
        bold=bool(style["bold"]),
        horizontal_align=str(style["horizontal_align"]),
        vertical_anchor=str(style["vertical_anchor"]),
        text_block_ids=[block.id for block in overlapping],
        margin_left=style.get("margin_left"),
        margin_right=style.get("margin_right"),
        margin_top=style.get("margin_top"),
    )


def _collect_page_spans(page: fitz.Page) -> list[dict[str, Any]]:
    """Flatten PDF text spans with geometry for per-cell style lookup."""

    spans: list[dict[str, Any]] = []
    try:
        text_dict = page.get_text("dict")
    except Exception as error:  # noqa: BLE001
        logger.debug("get_text(dict) failed: %s", error)
        return spans
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = str(span.get("text", ""))
                if not text.strip():
                    continue
                bbox_values = span.get("bbox")
                if not bbox_values or len(bbox_values) < 4:
                    continue
                bbox = (
                    float(bbox_values[0]),
                    float(bbox_values[1]),
                    float(bbox_values[2]),
                    float(bbox_values[3]),
                )
                font_name = str(span.get("font", "") or "")
                color = span.get("color")
                font_color = None
                if isinstance(color, int):
                    font_color = f"#{color & 0xFFFFFF:06X}"
                spans.append(
                    {
                        "bbox": bbox,
                        "text": text,
                        "size": float(span.get("size", 0.0) or 0.0),
                        "font_color": font_color,
                        "font_name": font_name or None,
                        "bold": "bold" in font_name.lower(),
                    }
                )
    return spans


def map_pdf_font_to_pptx(font_name: str | None) -> str | None:
    """Map common PDF Base-14 / CJK faces to PowerPoint-safe typefaces."""

    if not font_name:
        return None
    lower = font_name.lower()
    if lower.startswith("helv") or "helvetica" in lower or lower in {"arial", "sans"}:
        return "Arial"
    if lower.startswith("timo") or "times" in lower:
        return "Times New Roman"
    if lower.startswith("cour") or "courier" in lower:
        return "Courier New"
    return font_name


def _style_from_spans(cell_bbox: BBox, span_index: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive font size/color/bold/alignment from spans whose centers fall in the cell."""

    matched: list[dict[str, Any]] = []
    for span in span_index:
        bbox = span["bbox"]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if cell_bbox[0] <= cx <= cell_bbox[2] and cell_bbox[1] <= cy <= cell_bbox[3]:
            matched.append(span)
        elif intersection_ratio(bbox, cell_bbox) >= 0.5:
            matched.append(span)

    if not matched:
        return {
            "font_size": None,
            "font_color": None,
            "font_name": None,
            "bold": False,
            "horizontal_align": "left",
            "vertical_anchor": "middle",
            "margin_left": None,
            "margin_right": None,
            "margin_top": None,
        }

    sizes = [float(span["size"]) for span in matched if span["size"] > 0]
    colors = [span["font_color"] for span in matched if span["font_color"]]
    fonts = [span["font_name"] for span in matched if span.get("font_name")]
    bold = any(bool(span["bold"]) for span in matched)

    # Union of matched span boxes for alignment heuristics.
    text_x0 = min(span["bbox"][0] for span in matched)
    text_x1 = max(span["bbox"][2] for span in matched)
    text_y0 = min(span["bbox"][1] for span in matched)
    text_y1 = max(span["bbox"][3] for span in matched)
    cell_width = max(1.0, cell_bbox[2] - cell_bbox[0])
    cell_height = max(1.0, cell_bbox[3] - cell_bbox[1])
    left_gap = text_x0 - cell_bbox[0]
    right_gap = cell_bbox[2] - text_x1
    top_gap = text_y0 - cell_bbox[1]
    bottom_gap = cell_bbox[3] - text_y1

    if abs(left_gap - right_gap) <= cell_width * 0.12 and left_gap > cell_width * 0.08:
        horizontal_align = "center"
    elif right_gap < left_gap * 0.45 and right_gap < cell_width * 0.12:
        horizontal_align = "right"
    else:
        horizontal_align = "left"

    # Tall cells: only force top when the first line is packed against the top
    # edge. Balanced gaps (merged headers, centred labels) stay middle — do not
    # skip the centring test for tall cells.
    line_heights = [max(1.0, span["bbox"][3] - span["bbox"][1]) for span in matched]
    median_line = float(median(line_heights)) if line_heights else 12.0
    tall_cell = cell_height >= max(3.0 * median_line, 72.0)
    if tall_cell and top_gap <= max(median_line * 0.85, cell_height * 0.12):
        vertical_anchor = "top"
    elif abs(top_gap - bottom_gap) <= cell_height * 0.35:
        vertical_anchor = "middle"
    elif top_gap + cell_height * 0.1 < bottom_gap and top_gap < cell_height * 0.12:
        vertical_anchor = "top"
    elif bottom_gap + cell_height * 0.1 < top_gap and bottom_gap < cell_height * 0.12:
        vertical_anchor = "bottom"
    else:
        vertical_anchor = "middle"

    # Source insets for PPTX cell margins (Office default ~7pt when unknown).
    # Only set the margin on the alignment side so centred cells are not pushed
    # by a capped left_gap (12pt vs ~7.2pt default).
    margin_left = None
    margin_right = None
    if horizontal_align == "left":
        margin_left = max(0.0, min(12.0, left_gap))
    elif horizontal_align == "right":
        margin_right = max(0.0, min(12.0, right_gap))
    margin_top = max(0.0, min(12.0, top_gap))

    return {
        "font_size": float(median(sizes)) if sizes else None,
        "font_color": Counter(colors).most_common(1)[0][0] if colors else None,
        "font_name": map_pdf_font_to_pptx(
            Counter(fonts).most_common(1)[0][0] if fonts else None
        ),
        "bold": bold,
        "horizontal_align": horizontal_align,
        "vertical_anchor": vertical_anchor,
        "margin_left": margin_left,
        "margin_right": margin_right,
        "margin_top": margin_top,
    }


def _infer_table_border_style(
    page: fitz.Page,
    table_bbox: BBox,
) -> tuple[str | None, float | None, float | None, float | None]:
    """Stroke color plus outer/inner widths from ruled lines near the table."""

    colors: list[str] = []
    outer_widths: list[float] = []
    inner_widths: list[float] = []
    all_widths: list[float] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return None, None, None, None

    pad = 2.0
    outer = (
        table_bbox[0] - pad,
        table_bbox[1] - pad,
        table_bbox[2] + pad,
        table_bbox[3] + pad,
    )
    for path in drawings:
        if path.get("type") not in {"s", "fs"}:
            continue
        items = path.get("items") or []
        if not items or items[0][0] not in {"l", "re"}:
            continue
        color = rgb_tuple_to_hex(path.get("color"))
        width = path.get("width")
        if color is None or width is None:
            continue
        width_f = float(width)
        if width_f > 4.0:
            continue
        rect = path.get("rect")
        if rect is None:
            continue
        line_bbox = _rect_to_bbox(rect)
        if not _bbox_center_inside(line_bbox, outer, padding=pad):
            continue
        colors.append(color)
        all_widths.append(width_f)
        # Outer frame: segment hugs the table perimeter.
        on_left = abs(line_bbox[0] - table_bbox[0]) <= pad and abs(line_bbox[2] - table_bbox[0]) <= pad
        on_right = abs(line_bbox[0] - table_bbox[2]) <= pad and abs(line_bbox[2] - table_bbox[2]) <= pad
        on_top = abs(line_bbox[1] - table_bbox[1]) <= pad and abs(line_bbox[3] - table_bbox[1]) <= pad
        on_bottom = abs(line_bbox[1] - table_bbox[3]) <= pad and abs(line_bbox[3] - table_bbox[3]) <= pad
        if on_left or on_right or on_top or on_bottom:
            outer_widths.append(width_f)
        else:
            inner_widths.append(width_f)

    if not colors:
        return None, None, None, None
    border_color = Counter(colors).most_common(1)[0][0]
    border_width = float(median(all_widths))
    outer_width = float(median(outer_widths)) if outer_widths else border_width
    inner_width = float(median(inner_widths)) if inner_widths else border_width
    return border_color, border_width, outer_width, inner_width


def _estimate_table_geometry(
    rows: list[Any],
    bbox: BBox,
    row_count: int,
    col_count: int,
) -> tuple[list[float], list[float]]:
    """Derive column widths / row heights from cell rects.

    Prefer the union of all non-merged cell edges so a large empty / merged
    notes cell cannot skew neighbouring column widths or invent fractional
    row heights that shift subsequent text in PowerPoint.
    """

    row_heights: list[float] = []
    for row in rows:
        row_bbox = getattr(row, "bbox", None)
        if row_bbox is not None:
            rect = fitz.Rect(row_bbox)
            row_heights.append(max(1.0, float(rect.y1 - rect.y0)))
        else:
            row_heights.append(max(1.0, (bbox[3] - bbox[1]) / max(row_count, 1)))
    while len(row_heights) < row_count:
        row_heights.append(max(1.0, (bbox[3] - bbox[1]) / max(row_count, 1)))
    row_heights = row_heights[:row_count]

    # Collect left edges per column from non-None cell rects across rows.
    col_lefts: list[list[float]] = [[] for _ in range(col_count)]
    col_rights: list[list[float]] = [[] for _ in range(col_count)]
    for row in rows:
        row_cells = list(getattr(row, "cells", []) or [])
        while len(row_cells) < col_count:
            row_cells.append(None)
        for col_index in range(col_count):
            cell_rect = row_cells[col_index]
            if cell_rect is None:
                continue
            rect = fitz.Rect(cell_rect)
            # Skip clearly merged multi-column cells when inferring single-col width.
            width = float(rect.x1 - rect.x0)
            nominal = (bbox[2] - bbox[0]) / max(col_count, 1)
            if width > nominal * 1.6 and col_count > 1:
                continue
            col_lefts[col_index].append(float(rect.x0))
            col_rights[col_index].append(float(rect.x1))

    column_widths = [max(1.0, (bbox[2] - bbox[0]) / max(col_count, 1)) for _ in range(col_count)]
    if all(col_lefts[index] and col_rights[index] for index in range(col_count)):
        edges = [bbox[0]]
        for col_index in range(col_count - 1):
            right = float(median(col_rights[col_index]))
            left = float(median(col_lefts[col_index + 1]))
            edges.append((right + left) / 2.0)
        edges.append(bbox[2])
        column_widths = [max(1.0, edges[index + 1] - edges[index]) for index in range(col_count)]
    else:
        # Fallback: first row with a full set of non-merged cells.
        for row in rows:
            row_cells = [cell for cell in (getattr(row, "cells", []) or []) if cell is not None]
            if len(row_cells) >= col_count:
                widths = []
                for cell in row_cells[:col_count]:
                    rect = fitz.Rect(cell)
                    widths.append(max(1.0, float(rect.x1 - rect.x0)))
                if len(widths) == col_count:
                    column_widths = widths
                    break

    # Snap total width/height to the table bbox so empty trailing cells do not
    # accumulate float drift that shifts later text.
    width_sum = sum(column_widths) or 1.0
    target_width = max(1.0, bbox[2] - bbox[0])
    if abs(width_sum - target_width) > 0.5:
        column_widths = [width * target_width / width_sum for width in column_widths]
    height_sum = sum(row_heights) or 1.0
    target_height = max(1.0, bbox[3] - bbox[1])
    if abs(height_sum - target_height) > 0.5:
        row_heights = [height * target_height / height_sum for height in row_heights]
    return column_widths, row_heights


def apply_cell_fills(
    table: TableElement,
    page: fitz.Page,
    *,
    dpi: int = 72,
    image_boxes: list[BBox] | None = None,
) -> None:
    """Resolve solid cell fills without sampling centred glyphs.

    Preference order per cell:

    1. Keep an existing fill (e.g. zebra stripe assigned during inference).
    2. Opaque fill of the smallest filled vector rect containing the cell
       (inset 2pt). If that smallest rect is translucent, skip the vector step
       entirely (do not fall through to a larger opaque page fill).
    3. Modal colour of non-text raster pixels (4-bit quantised), when the mode
       covers ≥60% of remaining pixels — **skipped** when the table sits on an
       embedded photo so transparent cells keep the image visible.
    4. Median of non-text raster pixels (covers gradients / soft fills).
    """

    if not table.cells:
        return
    filled_rects = _collect_filled_vector_rects(page)
    word_boxes = _collect_word_boxes(page)
    images = list(image_boxes) if image_boxes is not None else _collect_image_boxes(page)
    table_over_image = table_overlaps_image(table.bbox, images)
    pixmap: Any | None = None
    scale = dpi / 72.0
    for cell in table.cells:
        cell_bbox = _cell_bbox_from_geometry(table, cell)
        if cell.fill_color:
            # Inferred zebra / prior fill — still prefer a tighter vector match.
            vector_fill = _opaque_vector_fill_for_cell(cell_bbox, filled_rects)
            if vector_fill is not None:
                cell.fill_color = vector_fill
            continue
        vector_fill = _opaque_vector_fill_for_cell(cell_bbox, filled_rects)
        if vector_fill is not None:
            # Ignore huge backdrop fills that are really the page / photo card
            # when the table is drawn over an image without per-cell paints.
            if table_over_image and not _vector_fill_looks_like_cell(
                cell_bbox, filled_rects, vector_fill
            ):
                cell.fill_color = None
                continue
            cell.fill_color = vector_fill
            continue
        if table_over_image:
            # Source cell has no fill — keep PPTX cell transparent so the photo
            # under the table remains visible.
            cell.fill_color = None
            continue
        if pixmap is None:
            try:
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            except Exception:  # noqa: BLE001
                return
        raster_fill = _raster_fill_for_cell(
            cell_bbox,
            pixmap=pixmap,
            scale=scale,
            word_boxes=word_boxes,
        )
        if raster_fill is not None:
            cell.fill_color = raster_fill

    # Borderless inferred tables with a header band (≤1 non-white fill colour):
    # drop near-white body fills so hairline row rules above the table stay
    # visible. Keep white cells on true zebra stacks (2+ non-white fills).
    if (
        table.provenance == "inferred_alignment_table"
        and (table.border_width or 0) <= 0
    ):
        non_white = {
            cell.fill_color
            for cell in table.cells
            if cell.fill_color and not _hex_is_near_white(cell.fill_color)
        }
        if len(non_white) <= 1:
            for cell in table.cells:
                if cell.fill_color and _hex_is_near_white(cell.fill_color):
                    cell.fill_color = None


def _hex_is_near_white(color: str, *, threshold: int = 245) -> bool:
    text = color.lstrip("#")
    if len(text) != 6:
        return False
    try:
        red = int(text[0:2], 16)
        green = int(text[2:4], 16)
        blue = int(text[4:6], 16)
    except ValueError:
        return False
    return red >= threshold and green >= threshold and blue >= threshold


def _collect_image_boxes(page: fitz.Page) -> list[BBox]:
    boxes: list[BBox] = []
    try:
        for image_info in page.get_image_info() or []:
            bbox = image_info.get("bbox")
            if bbox is None:
                continue
            boxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
    except Exception:  # noqa: BLE001
        return boxes
    return boxes


def table_overlaps_image(
    table_bbox: BBox,
    image_boxes: list[BBox] | None,
    *,
    min_ratio: float = 0.2,
) -> bool:
    """True when a meaningful fraction of the table sits on an embedded image."""

    if not image_boxes:
        return False
    table_area = bbox_area(table_bbox)
    if table_area <= 0:
        return False
    for image_bbox in image_boxes:
        overlap = intersection_area(table_bbox, image_bbox)
        if overlap / table_area >= min_ratio:
            return True
    return False


def table_should_erase_background(
    table: TableElement,
    image_boxes: list[BBox] | None = None,
) -> bool:
    """Whether the table region must be solid-filled out of a raster background.

    Transparent tables drawn over photos must leave the photo pixels in place;
    ruled / filled tables still erase so native cells are not double-painted.
    """

    if not table_overlaps_image(table.bbox, image_boxes):
        return True
    return any(bool(cell.fill_color) for cell in table.cells)


def transparent_table_text_mask_blocks(
    tables: list[TableElement],
    page: fitz.Page,
    image_boxes: list[BBox] | None = None,
    *,
    pad_pt: float = 1.5,
) -> list[TextBlock]:
    """Word boxes to inpaint when a transparent table sits on a photo.

    The full table bbox is not erased (so the photo remains), but glyphs that
    were absorbed into native cells must still leave the raster or they ghost
    under the editable PPTX text.
    """

    blocks: list[TextBlock] = []
    word_boxes = _collect_word_boxes(page)
    index = 0
    for table in tables:
        if table_should_erase_background(table, image_boxes):
            continue
        for word in word_boxes:
            cx = (word[0] + word[2]) / 2.0
            cy = (word[1] + word[3]) / 2.0
            if not (table.bbox[0] <= cx <= table.bbox[2] and table.bbox[1] <= cy <= table.bbox[3]):
                continue
            padded = (
                word[0] - pad_pt,
                word[1] - pad_pt,
                word[2] + pad_pt,
                word[3] + pad_pt,
            )
            blocks.append(
                TextBlock(
                    id=f"table_word_mask_{table.id}_{index}",
                    source="native",
                    bbox=padded,
                    text="",
                    confidence=1.0,
                )
            )
            index += 1
    return blocks


def _vector_fill_looks_like_cell(
    cell_bbox: BBox,
    filled_rects: list[tuple[BBox, str, float]],
    fill_hex: str,
) -> bool:
    """True when the chosen vector fill is cell/row sized, not a page backdrop."""

    cell_area = max(bbox_area(cell_bbox), 1.0)
    for rect_bbox, color, _opacity in filled_rects:
        if color != fill_hex:
            continue
        if intersection_ratio(cell_bbox, rect_bbox) < 0.5:
            continue
        rect_area = bbox_area(rect_bbox)
        # Row stripe or single cell: at most ~12× the cell (full table width).
        if rect_area <= cell_area * 12.0:
            return True
    return False


def apply_header_cell_fills(
    table: TableElement,
    page: fitz.Page,
    *,
    dpi: int = 72,
    image_boxes: list[BBox] | None = None,
) -> None:
    """Backwards-compatible alias for :func:`apply_cell_fills`."""

    apply_cell_fills(table, page, dpi=dpi, image_boxes=image_boxes)


def _cell_bbox_from_geometry(table: TableElement, cell: TableCell) -> BBox:
    x0 = table.bbox[0] + sum(table.column_widths[: cell.col])
    y0 = table.bbox[1] + sum(table.row_heights[: cell.row])
    x1 = x0 + sum(table.column_widths[cell.col : cell.col + cell.colspan])
    y1 = y0 + sum(table.row_heights[cell.row : cell.row + cell.rowspan])
    return (x0, y0, x1, y1)


def _collect_filled_vector_rects(page: fitz.Page) -> list[tuple[BBox, str, float]]:
    """Filled drawing rects as ``(bbox, #RRGGBB, opacity)``."""

    results: list[tuple[BBox, str, float]] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return results
    for path in drawings:
        fill = path.get("fill")
        if fill is None:
            continue
        fill_hex = rgb_tuple_to_hex(fill)
        if fill_hex is None:
            continue
        rect = path.get("rect")
        if rect is None:
            continue
        raw_opacity = path.get("fill_opacity")
        opacity = float(raw_opacity) if isinstance(raw_opacity, (int, float)) else 1.0
        results.append((_rect_to_bbox(rect), fill_hex, max(0.0, min(1.0, opacity))))
    return results


def _opaque_vector_fill_for_cell(
    cell_bbox: BBox,
    filled_rects: list[tuple[BBox, str, float]],
    *,
    inset_pt: float = CELL_FILL_VECTOR_INSET_PT,
    opaque_opacity: float = CELL_FILL_OPAQUE_OPACITY,
) -> str | None:
    """Opaque fill of the smallest containing vector rect, or ``None``.

    When the smallest containing rect is translucent, returns ``None`` so the
    caller samples rendered pixels instead of picking a larger opaque backdrop.
    """

    inner = (
        cell_bbox[0] + inset_pt,
        cell_bbox[1] + inset_pt,
        cell_bbox[2] - inset_pt,
        cell_bbox[3] - inset_pt,
    )
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        inner = cell_bbox
    best_area: float | None = None
    best_fill: str | None = None
    best_opacity = 1.0
    for rect_bbox, fill_hex, opacity in filled_rects:
        if not (
            rect_bbox[0] <= inner[0]
            and rect_bbox[1] <= inner[1]
            and rect_bbox[2] >= inner[2]
            and rect_bbox[3] >= inner[3]
        ):
            continue
        # Skip hairline gridline rects — they are rules, not cell fills.
        width = abs(rect_bbox[2] - rect_bbox[0])
        height = abs(rect_bbox[3] - rect_bbox[1])
        if min(width, height) <= HAIRLINE_RULE_MAX_SHORT_PT:
            continue
        area = bbox_area(rect_bbox)
        if best_area is None or area < best_area:
            best_area = area
            best_fill = fill_hex
            best_opacity = opacity
    if best_fill is None:
        return None
    if best_opacity < opaque_opacity:
        return None
    return best_fill


def _collect_word_boxes(page: fitz.Page) -> list[BBox]:
    boxes: list[BBox] = []
    try:
        words = page.get_text("words") or []
    except Exception:  # noqa: BLE001
        return boxes
    for word in words:
        if not isinstance(word, (tuple, list)) or len(word) < 4:
            continue
        try:
            boxes.append((float(word[0]), float(word[1]), float(word[2]), float(word[3])))
        except (TypeError, ValueError):
            continue
    return boxes


def _non_text_pixels_for_cell(
    cell_bbox: BBox,
    *,
    pixmap: Any,
    scale: float,
    word_boxes: list[BBox],
) -> np.ndarray | None:
    """RGB pixels inside ``cell_bbox`` with word boxes masked out."""

    left = max(0, int(cell_bbox[0] * scale))
    top = max(0, int(cell_bbox[1] * scale))
    right = min(pixmap.width, int(math.ceil(cell_bbox[2] * scale)))
    bottom = min(pixmap.height, int(math.ceil(cell_bbox[3] * scale)))
    if right <= left or bottom <= top:
        return None

    samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
    n = int(pixmap.n)
    if n < 3:
        return None
    pixels = samples.reshape(pixmap.height, pixmap.width, n)[top:bottom, left:right, :3]
    if pixels.size == 0:
        return None

    mask = np.ones(pixels.shape[:2], dtype=bool)
    for word_bbox in word_boxes:
        if (
            word_bbox[2] <= cell_bbox[0]
            or word_bbox[0] >= cell_bbox[2]
            or word_bbox[3] <= cell_bbox[1]
            or word_bbox[1] >= cell_bbox[3]
        ):
            continue
        w_left = max(left, int(word_bbox[0] * scale))
        w_top = max(top, int(word_bbox[1] * scale))
        w_right = min(right, int(math.ceil(word_bbox[2] * scale)))
        w_bottom = min(bottom, int(math.ceil(word_bbox[3] * scale)))
        if w_right > w_left and w_bottom > w_top:
            mask[w_top - top : w_bottom - top, w_left - left : w_right - left] = False

    remaining = pixels[mask]
    if remaining.size == 0:
        return None
    return remaining


def _quantize_rgb(pixels: np.ndarray, *, bits: int = CELL_FILL_QUANTIZE_BITS) -> np.ndarray:
    """Round each channel to ``bits`` bits (reduces JPEG / AA noise for modal tests)."""

    shift = max(0, 8 - bits)
    if shift == 0:
        return pixels
    # Floor to the bin (avoid ``255 + round`` overflowing past 255 → 0).
    return (pixels.astype(np.uint16) >> shift << shift).astype(np.uint8)


def _rgb_to_hex(rgb: tuple[int, int, int] | np.ndarray) -> str:
    channels = [int(max(0, min(255, round(float(channel))))) for channel in rgb[:3]]
    return f"#{channels[0]:02X}{channels[1]:02X}{channels[2]:02X}"


def _raster_fill_for_cell(
    cell_bbox: BBox,
    *,
    pixmap: Any,
    scale: float,
    word_boxes: list[BBox],
    min_share: float = CELL_FILL_MODAL_MIN_SHARE,
) -> str | None:
    """Modal (quantised) non-text colour, else median — covers soft / gradient fills."""

    remaining = _non_text_pixels_for_cell(
        cell_bbox,
        pixmap=pixmap,
        scale=scale,
        word_boxes=word_boxes,
    )
    if remaining is None:
        return None

    quantised = _quantize_rgb(remaining)
    packed = (
        quantised[:, 0].astype(np.int32) << 16
        | quantised[:, 1].astype(np.int32) << 8
        | quantised[:, 2].astype(np.int32)
    )
    values, counts = np.unique(packed, return_counts=True)
    best_index = int(np.argmax(counts))
    share = float(counts[best_index]) / float(packed.size)
    if share >= min_share:
        # Quantisation only groups noise; emit the median of the original pixels
        # in the winning bucket so translucent composites stay accurate.
        bucket = remaining[packed == values[best_index]]
        return _rgb_to_hex(np.median(bucket.astype(np.float64), axis=0))

    # Gradient / soft fills: no single bucket dominates — use median RGB.
    median_rgb = np.median(remaining.astype(np.float64), axis=0)
    return _rgb_to_hex(median_rgb)


# Backwards-compatible name used by earlier unit tests / call sites.
def _modal_fill_for_cell(
    cell_bbox: BBox,
    *,
    pixmap: Any,
    scale: float,
    word_boxes: list[BBox],
    min_share: float = CELL_FILL_MODAL_MIN_SHARE,
) -> str | None:
    return _raster_fill_for_cell(
        cell_bbox,
        pixmap=pixmap,
        scale=scale,
        word_boxes=word_boxes,
        min_share=min_share,
    )


__all__ = [
    "CELL_FILL_MODAL_MIN_SHARE",
    "CELL_FILL_OPAQUE_OPACITY",
    "CELL_FILL_QUANTIZE_BITS",
    "CELL_FILL_VECTOR_INSET_PT",
    "HAIRLINE_RULE_MAX_SHORT_PT",
    "HAIRLINE_RULE_MIN_LONG_PT",
    "LARGE_EMPTY_CELL_PAGE_RATIO",
    "PAGE_BACKGROUND_AREA_RATIO",
    "SPURIOUS_TABLE_MAX_CELLS",
    "SPURIOUS_TABLE_PAGE_RATIO",
    "VectorExtractionResult",
    "apply_cell_fills",
    "apply_header_cell_fills",
    "bbox_area",
    "extract_tables",
    "extract_vector_content",
    "filter_text_blocks_outside_tables",
    "is_page_background_fill",
    "parse_dash_style",
    "split_page_background_stroke",
    "remaining_visuals_after_nativization",
    "resolve_complexity_limits",
    "rgb_tuple_to_hex",
    "should_attempt_vector_nativization",
    "table_overlaps_image",
    "table_should_erase_background",
    "transparent_table_text_mask_blocks",
]
