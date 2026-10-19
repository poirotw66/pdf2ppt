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

# Full-page fills at or above this fraction become page-background shapes.
PAGE_BACKGROUND_AREA_RATIO = 0.92
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
        if _drawing_belongs_to_table(path, table_bboxes, page_area=page_area):
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
        shapes.append(shape)
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


def extract_tables(
    page: fitz.Page,
    text_blocks: list[TextBlock],
    *,
    page_number: int,
) -> list[TableElement]:
    """Detect ruled tables and map cell text / style from native text blocks."""

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
    if finder is None:
        return []

    span_index = _collect_page_spans(page)
    border_color, border_width = _infer_table_border_style(page)
    tables: list[TableElement] = []
    for table_index, table in enumerate(getattr(finder, "tables", []) or []):
        element = _table_to_element(
            table,
            text_blocks=text_blocks,
            page_number=page_number,
            table_index=table_index,
            span_index=span_index,
            border_color=border_color,
            border_width=border_width,
        )
        if element is not None:
            tables.append(element)
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


def _drawing_belongs_to_table(path: dict[str, Any], table_bboxes: list[BBox], *, page_area: float) -> bool:
    if not table_bboxes:
        return False
    rect = path.get("rect")
    if rect is None:
        return False
    bbox = _rect_to_bbox(rect)
    # Never absorb a page-sized fill into a table.
    if bbox_area(bbox) / page_area >= PAGE_BACKGROUND_AREA_RATIO:
        return False
    for table_bbox in table_bboxes:
        if intersection_ratio(bbox, table_bbox) >= TABLE_DRAWING_CONTAINMENT_RATIO:
            return True
        # Horizontal/vertical rules often have zero area; test the segment midpoint.
        if _bbox_center_inside(bbox, table_bbox, padding=1.0):
            return True
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

    is_page_background = bbox_area(bbox) / page_area >= PAGE_BACKGROUND_AREA_RATIO and fill is not None

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
    """Pick the MSO arrow facing the sharp tip of a closed arrow polygon."""

    if len(points) < 3:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return "right" if width >= height else "down"

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    # Tip is the vertex farthest from the bbox center along the long axis.
    tip = max(points, key=lambda point: math.hypot(point[0] - cx, point[1] - cy))
    dx = tip[0] - cx
    dy = tip[1] - cy
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def _table_to_element(
    table: Any,
    *,
    text_blocks: list[TextBlock],
    page_number: int,
    table_index: int,
    span_index: list[dict[str, Any]],
    border_color: str | None,
    border_width: float | None,
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
        z_order=10_000 + table_index,
        confidence=0.9,
    )


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
        bold=bool(style["bold"]),
        horizontal_align=str(style["horizontal_align"]),
        vertical_anchor=str(style["vertical_anchor"]),
        text_block_ids=[block.id for block in overlapping],
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
                        "bold": "bold" in font_name.lower(),
                    }
                )
    return spans


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
            "bold": False,
            "horizontal_align": "left",
            "vertical_anchor": "middle",
        }

    sizes = [float(span["size"]) for span in matched if span["size"] > 0]
    colors = [span["font_color"] for span in matched if span["font_color"]]
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

    if abs(top_gap - bottom_gap) <= cell_height * 0.2:
        vertical_anchor = "middle"
    elif top_gap < bottom_gap:
        vertical_anchor = "top"
    else:
        vertical_anchor = "bottom"

    return {
        "font_size": float(median(sizes)) if sizes else None,
        "font_color": Counter(colors).most_common(1)[0][0] if colors else None,
        "bold": bold,
        "horizontal_align": horizontal_align,
        "vertical_anchor": vertical_anchor,
    }


def _infer_table_border_style(page: fitz.Page) -> tuple[str | None, float | None]:
    """Most common stroke color/width among line drawings (table rules)."""

    colors: list[str] = []
    widths: list[float] = []
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return None, None
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
        # Prefer thin ruled lines over thick decorative strokes.
        if float(width) > 4.0:
            continue
        colors.append(color)
        widths.append(float(width))
    if not colors:
        return None, None
    return Counter(colors).most_common(1)[0][0], float(median(widths))


def _estimate_table_geometry(
    rows: list[Any],
    bbox: BBox,
    row_count: int,
    col_count: int,
) -> tuple[list[float], list[float]]:
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

    # Column widths from first row with complete cells.
    column_widths = [max(1.0, (bbox[2] - bbox[0]) / max(col_count, 1)) for _ in range(col_count)]
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
    return column_widths, row_heights


def apply_header_cell_fills(
    table: TableElement,
    page: fitz.Page,
    *,
    dpi: int = 72,
) -> None:
    """Sample cell center colors for header-row fills when available."""

    if not table.cells:
        return
    try:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    except Exception:  # noqa: BLE001
        return
    scale = dpi / 72.0
    samples = pixmap.samples
    width = pixmap.width
    n = pixmap.n
    for cell in table.cells:
        if cell.row != 0:
            continue
        # Locate cell bbox via equal split if needed.
        x0 = table.bbox[0] + sum(table.column_widths[: cell.col])
        y0 = table.bbox[1] + sum(table.row_heights[: cell.row])
        x1 = x0 + sum(table.column_widths[cell.col : cell.col + cell.colspan])
        y1 = y0 + sum(table.row_heights[cell.row : cell.row + cell.rowspan])
        cx = int(((x0 + x1) / 2.0) * scale)
        cy = int(((y0 + y1) / 2.0) * scale)
        if not (0 <= cx < width and 0 <= cy < pixmap.height):
            continue
        offset = (cy * width + cx) * n
        if offset + 2 >= len(samples):
            continue
        cell.fill_color = f"#{samples[offset]:02X}{samples[offset + 1]:02X}{samples[offset + 2]:02X}"


__all__ = [
    "PAGE_BACKGROUND_AREA_RATIO",
    "VectorExtractionResult",
    "apply_header_cell_fills",
    "bbox_area",
    "extract_tables",
    "extract_vector_content",
    "filter_text_blocks_outside_tables",
    "parse_dash_style",
    "remaining_visuals_after_nativization",
    "resolve_complexity_limits",
    "rgb_tuple_to_hex",
    "should_attempt_vector_nativization",
]
