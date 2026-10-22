"""Infer borderless / zebra-striped tables from aligned text and row fills.

PyMuPDF ``find_tables()`` (line strategy) misses grids with no stroked rules.
This module recovers those as native ``TableElement``s when column x-alignment
and row baselines (optionally plus alternating filled row rects) form a clear
grid, while rejecting multi-column body copy, bullet lists, and card layouts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from statistics import median
from typing import Any

import fitz

from .models import BBox, TableCell, TableElement, TextBlock
from .vector_nativization import (
    _build_table_cell,
    _collect_page_spans,
    bbox_area,
    intersection_area,
    intersection_ratio,
    rgb_tuple_to_hex,
)

logger = logging.getLogger(__name__)

# --- Detection thresholds ----------------------------------------------------
MIN_TABLE_ROWS = 3
MIN_TABLE_COLS = 2
MAX_TABLE_COLS = 10
# Word left-edge / baseline clustering.
COLUMN_X_TOL_PT = 12.0
ROW_Y_TOL_PT = 8.0
# Cell text must stay short (tables, not wrapped paragraphs).
MAX_MEDIAN_CELL_CHARS = 28
MAX_CELL_CHARS = 72
MIN_OCCUPANCY = 0.45
# Minimum horizontal gap between columns relative to median word width.
MIN_COL_GAP_RATIO = 1.2
# Row pitch coefficient of variation above this → reject (uneven article lines).
MAX_ROW_PITCH_CV = 0.55
# Zebra row bands: wide short fills with alternating colours.
ZEBRA_MIN_ROWS = 3
ZEBRA_MAX_ROW_HEIGHT_PT = 64.0
ZEBRA_MIN_ROW_HEIGHT_PT = 10.0
ZEBRA_MIN_WIDTH_OVER_HEIGHT = 2.5
ZEBRA_MAX_X_DRIFT_PT = 8.0
ZEBRA_MAX_GAP_PT = 6.0
# Bullet / list markers (standalone cell or leading a longer phrase).
_BULLET_RE = re.compile(
    r"^(\u2022|\u25CF|\u25A0|\u25E6|\u2219|•|●|■|◦|-|–|—|\*|\d+[\.\)])$"
)
_LIST_PREFIX_RE = re.compile(
    r"^(?:"
    r"[\u2022\u25CF\u25A0\u25E6\u2219•●■◦\-–—\*]"
    r"|\d+[\.\)]"
    r"|[a-zA-Z][\.\)]"
    r")\s+"
)
# Cap window search so dense pages cannot hang CI (O(rows²) otherwise).
MAX_INFERENCE_ROWS = 48
MAX_WINDOW_ROWS = 24
# Card-layout: large filled rects that each contain a text cluster.
CARD_MIN_AREA_PT2 = 8_000.0
CARD_CONTAINMENT_RATIO = 0.55


@dataclass(slots=True)
class _Word:
    bbox: BBox
    text: str

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


@dataclass(slots=True)
class _ZebraBand:
    bbox: BBox
    fill_color: str


def detect_inferred_tables(
    page: fitz.Page,
    text_blocks: list[TextBlock],
    *,
    page_number: int,
    existing_table_bboxes: list[BBox] | None = None,
) -> list[TableElement]:
    """Detect borderless / zebra tables not already covered by ruled find_tables()."""

    existing = list(existing_table_bboxes or [])
    words = _collect_words(page)
    if len(words) < MIN_TABLE_ROWS * MIN_TABLE_COLS:
        return []

    zebra_bands = _detect_zebra_bands(page)
    candidates: list[TableElement] = []

    # Prefer zebra-backed grids first so title/caption text cannot claim the
    # same region as a looser alignment-only candidate.
    for index, zebra_grid in enumerate(_iter_zebra_grid_candidates(zebra_bands, words)):
        if _overlaps_existing(zebra_grid["bbox"], existing):
            continue
        if not _passes_text_grid_guards(zebra_grid, words=words, page=page, allow_zebra=True):
            continue
        element = _grid_to_table_element(
            zebra_grid,
            text_blocks=text_blocks,
            page=page,
            page_number=page_number,
            table_index=20_000 + index,
            zebra_bands=zebra_bands,
            borderless=True,
        )
        if element is not None:
            candidates.append(element)
            existing.append(element.bbox)

    for index, grid in enumerate(_iter_text_grid_candidates(words)):
        if _overlaps_existing(grid["bbox"], existing):
            continue
        if not _passes_text_grid_guards(grid, words=words, page=page):
            continue
        element = _grid_to_table_element(
            grid,
            text_blocks=text_blocks,
            page=page,
            page_number=page_number,
            table_index=10_000 + index,
            zebra_bands=zebra_bands,
            borderless=True,
        )
        if element is not None:
            candidates.append(element)
            existing.append(element.bbox)

    return candidates


def _collect_words(page: fitz.Page) -> list[_Word]:
    words: list[_Word] = []
    try:
        raw = page.get_text("words") or []
    except Exception:  # noqa: BLE001
        return words
    for entry in raw:
        if not isinstance(entry, (tuple, list)) or len(entry) < 5:
            continue
        text = str(entry[4]).strip()
        if not text:
            continue
        try:
            bbox = (float(entry[0]), float(entry[1]), float(entry[2]), float(entry[3]))
        except (TypeError, ValueError):
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        words.append(_Word(bbox=bbox, text=text))
    return words


def _cluster_sorted(values: list[float], *, tol: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= tol:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(sum(cluster) / len(cluster)) for cluster in clusters]


def _assign_cluster(value: float, centers: list[float], *, tol: float) -> int | None:
    best_index: int | None = None
    best_dist = tol
    for index, center in enumerate(centers):
        dist = abs(value - center)
        if dist <= best_dist:
            best_dist = dist
            best_index = index
    return best_index


def _assign_column_index(x0: float, col_centers: list[float], *, tol: float) -> int:
    """Map a word left edge to a column.

    Uses the rightmost column whose left edge is at or left of ``x0`` (with
    tolerance). This keeps subsequent words in a multi-word cell
    (``Item 1``) in the same column instead of failing nearest-center match.
    """

    best = 0
    for index, center in enumerate(col_centers):
        if center <= x0 + tol:
            best = index
        else:
            break
    return best


def _iter_text_grid_candidates(words: list[_Word]) -> list[dict[str, Any]]:
    if not words:
        return []
    heights = [word.y1 - word.y0 for word in words]
    row_tol = max(ROW_Y_TOL_PT, float(median(heights)) * 0.7)
    row_centers = _cluster_sorted([word.cy for word in words], tol=row_tol)
    if len(row_centers) < MIN_TABLE_ROWS:
        return []

    rows: list[list[_Word]] = [[] for _ in row_centers]
    for word in words:
        index = _assign_cluster(word.cy, row_centers, tol=row_tol)
        if index is not None:
            rows[index].append(word)
    rows = [row for row in rows if row]
    if len(rows) < MIN_TABLE_ROWS:
        return []

    # Grow maximal windows, then trim leading/trailing title/caption rows.
    # Full O(n²) enumeration hangs slide-quality CI on dense pages.
    if len(rows) > MAX_INFERENCE_ROWS:
        rows = rows[:MAX_INFERENCE_ROWS]
    candidates: list[dict[str, Any]] = []
    start = 0
    while start < len(rows):
        best: dict[str, Any] | None = None
        best_end = start
        max_end = min(len(rows), start + MAX_WINDOW_ROWS)
        for end in range(start + MIN_TABLE_ROWS, max_end + 1):
            grid = _try_build_grid(rows[start:end])
            if grid is None:
                if best is not None:
                    break
                continue
            best = grid
            best_end = end
        if best is None:
            start += 1
            continue
        candidates.append(best)
        # Trim up to two rows from each end to drop glued titles/captions.
        for trim_lo in range(0, 3):
            for trim_hi in range(0, 3):
                if trim_lo == 0 and trim_hi == 0:
                    continue
                inner_start = start + trim_lo
                inner_end = best_end - trim_hi
                if inner_end - inner_start < MIN_TABLE_ROWS:
                    continue
                trimmed = _try_build_grid(rows[inner_start:inner_end])
                if trimmed is not None:
                    candidates.append(trimmed)
        start = max(start + 1, best_end - 1)
    return _dedupe_grids(candidates)


def _try_build_grid(rows: list[list[_Word]]) -> dict[str, Any] | None:
    if len(rows) < MIN_TABLE_ROWS:
        return None
    widths = [word.x1 - word.x0 for row in rows for word in row]
    if not widths:
        return None
    median_width = max(1.0, float(median(widths)))
    # Column breaks from large intra-row gaps (not every word left edge).
    gap_threshold = max(COLUMN_X_TOL_PT, median_width * MIN_COL_GAP_RATIO)
    col_votes: list[float] = []
    for row in rows:
        ordered = sorted(row, key=lambda word: word.x0)
        col_votes.append(ordered[0].x0)
        for left, right in zip(ordered, ordered[1:]):
            gap = right.x0 - left.x1
            if gap >= gap_threshold:
                col_votes.append(right.x0)
    col_centers = _cluster_sorted(col_votes, tol=COLUMN_X_TOL_PT)
    if not (MIN_TABLE_COLS <= len(col_centers) <= MAX_TABLE_COLS):
        return None

    # Every row must map to the same column count (missing cells allowed).
    matrix: list[list[str]] = []
    cell_words: list[list[list[_Word]]] = []
    for row in rows:
        buckets: list[list[_Word]] = [[] for _ in col_centers]
        for word in row:
            index = _assign_column_index(word.x0, col_centers, tol=COLUMN_X_TOL_PT)
            buckets[index].append(word)
        # Reject rows that collapse to a single occupied column (paragraph line).
        occupied = sum(1 for bucket in buckets if bucket)
        if occupied < MIN_TABLE_COLS:
            return None
        texts = [" ".join(word.text for word in sorted(bucket, key=lambda item: item.x0)) for bucket in buckets]
        matrix.append(texts)
        cell_words.append(buckets)

    col_count = len(col_centers)
    row_count = len(matrix)
    # Geometry from word extents; pad so fills that hug text stay inside.
    all_words = [word for row in rows for word in row]
    x_min = min(word.x0 for word in all_words)
    x_max = max(word.x1 for word in all_words)
    y_min = min(word.y0 for word in all_words)
    y_max = max(word.y1 for word in all_words)
    pad = 4.0
    bbox: BBox = (x_min - pad, y_min - pad * 0.5, x_max + pad, y_max + pad * 0.5)

    # Column edges from each column's text extent: midpoint between previous
    # column's max-x1 and next column's min-x0 (not midpoint of left edges).
    col_x0: list[float] = []
    col_x1: list[float] = []
    for col_index in range(col_count):
        col_words = [word for buckets in cell_words for word in buckets[col_index]]
        if not col_words:
            col_x0.append(col_centers[col_index])
            next_center = col_centers[col_index + 1] if col_index + 1 < col_count else bbox[2]
            col_x1.append(next_center)
            continue
        col_x0.append(min(word.x0 for word in col_words))
        col_x1.append(max(word.x1 for word in col_words))
    edges = [bbox[0]]
    for col_index in range(col_count - 1):
        gap_mid = (col_x1[col_index] + col_x0[col_index + 1]) / 2.0
        # Prefer next-col left minus pad when the gap is wide (left-aligned grids).
        next_left_inset = col_x0[col_index + 1] - pad
        if next_left_inset > col_x1[col_index]:
            edges.append(max(col_x1[col_index], next_left_inset))
        else:
            edges.append(gap_mid)
    edges.append(bbox[2])
    column_widths = [max(1.0, edges[index + 1] - edges[index]) for index in range(col_count)]

    # Majority horizontal alignment per column (avoids single-cell center/right).
    column_aligns: list[str] = []
    for col_index in range(col_count):
        votes: list[str] = []
        cell_left = edges[col_index]
        cell_right = edges[col_index + 1]
        cell_width = max(1.0, cell_right - cell_left)
        for buckets in cell_words:
            words_in_cell = buckets[col_index]
            if not words_in_cell:
                continue
            text_x0 = min(word.x0 for word in words_in_cell)
            text_x1 = max(word.x1 for word in words_in_cell)
            left_gap = text_x0 - cell_left
            right_gap = cell_right - text_x1
            if abs(left_gap - right_gap) <= cell_width * 0.12 and left_gap > cell_width * 0.08:
                votes.append("center")
            elif right_gap < left_gap * 0.45 and right_gap < cell_width * 0.12:
                votes.append("right")
            else:
                votes.append("left")
        if not votes:
            column_aligns.append("left")
        else:
            column_aligns.append(max(set(votes), key=votes.count))

    # Use midpoints between successive row tops for interior boundaries.
    boundaries = [bbox[1]]
    for index in range(len(rows) - 1):
        row_bottom = max(word.y1 for word in rows[index])
        next_top = min(word.y0 for word in rows[index + 1])
        boundaries.append((row_bottom + next_top) / 2.0)
    boundaries.append(bbox[3])
    row_heights = [max(1.0, boundaries[index + 1] - boundaries[index]) for index in range(row_count)]

    return {
        "bbox": bbox,
        "row_count": row_count,
        "col_count": col_count,
        "matrix": matrix,
        "cell_words": cell_words,
        "column_widths": column_widths,
        "row_heights": row_heights,
        "column_aligns": column_aligns,
        "row_centers": [float(sum(word.cy for word in row) / len(row)) for row in rows],
    }


def _dedupe_grids(grids: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _score(grid: dict[str, Any]) -> tuple[float, int, int]:
        matrix = grid["matrix"]
        nonempty = sum(1 for row in matrix for cell in row if str(cell).strip())
        occupancy = nonempty / max(1, int(grid["row_count"]) * int(grid["col_count"]))
        return (occupancy, nonempty, int(grid["row_count"]) * int(grid["col_count"]))

    kept: list[dict[str, Any]] = []
    for grid in sorted(grids, key=lambda item: (-_score(item)[0], -_score(item)[1], item["bbox"][1])):
        if any(intersection_ratio(grid["bbox"], other["bbox"]) >= 0.6 for other in kept):
            continue
        kept.append(grid)
    return kept


def _detect_zebra_bands(page: fitz.Page) -> list[_ZebraBand]:
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return []
    bands: list[_ZebraBand] = []
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
        bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if height < ZEBRA_MIN_ROW_HEIGHT_PT or height > ZEBRA_MAX_ROW_HEIGHT_PT:
            continue
        if width < height * ZEBRA_MIN_WIDTH_OVER_HEIGHT:
            continue
        # Skip near-page fills.
        page_area = max(float(page.rect.width * page.rect.height), 1.0)
        if bbox_area(bbox) / page_area >= 0.5:
            continue
        bands.append(_ZebraBand(bbox=bbox, fill_color=fill_hex))
    bands.sort(key=lambda band: (band.bbox[1], band.bbox[0]))
    return bands


def _iter_zebra_grid_candidates(
    bands: list[_ZebraBand],
    words: list[_Word],
) -> list[dict[str, Any]]:
    if len(bands) < ZEBRA_MIN_ROWS:
        return []
    groups: list[list[_ZebraBand]] = []
    current: list[_ZebraBand] = []
    for band in bands:
        if not current:
            current = [band]
            continue
        prev = current[-1]
        same_width = (
            abs(band.bbox[0] - prev.bbox[0]) <= ZEBRA_MAX_X_DRIFT_PT
            and abs(band.bbox[2] - prev.bbox[2]) <= ZEBRA_MAX_X_DRIFT_PT
        )
        gap = band.bbox[1] - prev.bbox[3]
        if same_width and -2.0 <= gap <= ZEBRA_MAX_GAP_PT:
            current.append(band)
        else:
            if len(current) >= ZEBRA_MIN_ROWS:
                groups.append(current)
            current = [band]
    if len(current) >= ZEBRA_MIN_ROWS:
        groups.append(current)

    grids: list[dict[str, Any]] = []
    for group in groups:
        colors = [band.fill_color for band in group]
        if len(set(colors)) < 2:
            continue
        # Alternation: neighbouring rows differ for most adjacent pairs.
        alternations = sum(1 for left, right in zip(colors, colors[1:]) if left != right)
        if alternations < max(1, len(group) - 2):
            continue
        x0 = min(band.bbox[0] for band in group)
        y0 = group[0].bbox[1]
        x1 = max(band.bbox[2] for band in group)
        y1 = group[-1].bbox[3]
        bbox: BBox = (x0, y0, x1, y1)
        inside = [
            word
            for word in words
            if word.cx >= x0 and word.cx <= x1 and word.cy >= y0 and word.cy <= y1
        ]
        if len(inside) < MIN_TABLE_COLS * 2:
            continue
        # Build a text grid clipped to the zebra stack.
        sub_grids = _iter_text_grid_candidates(inside)
        if sub_grids:
            for grid in sub_grids:
                # Snap bbox / row heights to zebra bands when row counts match.
                if grid["row_count"] == len(group):
                    grid = dict(grid)
                    grid["bbox"] = bbox
                    grid["row_heights"] = [max(1.0, band.bbox[3] - band.bbox[1]) for band in group]
                    grid["zebra_fills"] = [band.fill_color for band in group]
                grids.append(grid)
            continue
        # Fallback: one row per band, columns from word x clustering.
        synthetic_rows: list[list[_Word]] = [[] for _ in group]
        for word in inside:
            for index, band in enumerate(group):
                if band.bbox[1] <= word.cy <= band.bbox[3]:
                    synthetic_rows[index].append(word)
                    break
        synthetic_rows = [row for row in synthetic_rows if row]
        built = _try_build_grid(synthetic_rows)
        if built is None:
            continue
        built["bbox"] = bbox
        built["row_heights"] = [
            max(1.0, band.bbox[3] - band.bbox[1]) for band in group[: int(built["row_count"])]
        ]
        built["zebra_fills"] = [band.fill_color for band in group[: int(built["row_count"])]]
        grids.append(built)
    return _dedupe_grids(grids)


def _passes_text_grid_guards(
    grid: dict[str, Any],
    *,
    words: list[_Word],
    page: fitz.Page,
    allow_zebra: bool = False,
) -> bool:
    matrix: list[list[str]] = grid["matrix"]
    row_count = int(grid["row_count"])
    col_count = int(grid["col_count"])
    if row_count < MIN_TABLE_ROWS or col_count < MIN_TABLE_COLS:
        return False

    flat = [cell.strip() for row in matrix for cell in row]
    nonempty = [cell for cell in flat if cell]
    if not nonempty:
        return False
    occupancy = len(nonempty) / max(1, row_count * col_count)
    if occupancy < MIN_OCCUPANCY:
        return False
    lengths = [len(cell) for cell in nonempty]
    if float(median(lengths)) > MAX_MEDIAN_CELL_CHARS:
        return False
    if max(lengths) > MAX_CELL_CHARS:
        return False

    # Bullet / numbered list: any column dominated by list markers or prefixes.
    for col_index in range(col_count):
        col_cells = [row[col_index].strip() for row in matrix if row[col_index].strip()]
        if not col_cells:
            continue
        bullet_hits = sum(
            1
            for cell in col_cells
            if _BULLET_RE.match(cell) or _LIST_PREFIX_RE.match(cell)
        )
        if bullet_hits / len(col_cells) >= 0.5:
            return False

    # Real tables almost always have at least one short/numeric column
    # (IDs, scores, codes). Two-column Strengths/Risks bullet lists do not.
    if not any(_column_is_short_or_numeric(matrix, col_index) for col_index in range(col_count)):
        return False

    # Multi-column article: two wide columns of long wrapping text.
    if col_count == 2 and float(median(lengths)) > 18 and occupancy > 0.85:
        cell_word_counts = [
            len(cell.split())
            for row in matrix
            for cell in row
            if cell.strip()
        ]
        if cell_word_counts and float(median(cell_word_counts)) >= 5:
            return False

    # Uneven row pitch → body copy, not a table (zebra bands may still be even).
    centers = list(grid.get("row_centers") or [])
    if len(centers) >= 3:
        pitches = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
        pitches = [pitch for pitch in pitches if pitch > 1.0]
        if pitches:
            mean_pitch = sum(pitches) / len(pitches)
            if mean_pitch > 0:
                variance = sum((pitch - mean_pitch) ** 2 for pitch in pitches) / len(pitches)
                cv = (variance**0.5) / mean_pitch
                if cv > MAX_ROW_PITCH_CV and not (allow_zebra and grid.get("zebra_fills")):
                    return False

    if _looks_like_card_layout(grid["bbox"], page):
        return False

    # Require at least one fully populated column (header-like consistency).
    col_fill = [
        sum(1 for row in matrix if row[col].strip()) / row_count for col in range(col_count)
    ]
    if max(col_fill) < 0.7:
        return False
    # Title / caption bleed often leaves a nearly empty left column while the
    # real grid columns are dense — reject that ragged shape.
    if min(col_fill) < 0.35 and max(col_fill) >= 0.7:
        return False
    # A single very long cell (wrapped caption) in an otherwise short grid.
    if any(len(cell) > 40 for cell in nonempty) and float(median(lengths)) <= 16:
        return False

    _ = words  # reserved for future density checks
    return True


def _column_is_short_or_numeric(matrix: list[list[str]], col_index: int) -> bool:
    """True when most nonempty cells in a column are short tokens or numbers."""

    cells = [row[col_index].strip() for row in matrix if row[col_index].strip()]
    if not cells:
        return False
    hits = 0
    for cell in cells:
        tokens = cell.split()
        if len(cell) <= 12 and len(tokens) <= 2:
            hits += 1
            continue
        if re.fullmatch(r"[\d.,:%$+\-]+", cell):
            hits += 1
    return hits / len(cells) >= 0.55


def _looks_like_card_layout(table_bbox: BBox, page: fitz.Page) -> bool:
    """True when text sits in separate large filled cards rather than one grid."""

    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return False
    cards: list[BBox] = []
    for path in drawings:
        if path.get("fill") is None:
            continue
        rect = path.get("rect")
        if rect is None:
            continue
        bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        if bbox_area(bbox) < CARD_MIN_AREA_PT2:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        # Cards are blocky, not thin zebra stripes.
        if height < 80 or width < 80:
            continue
        if intersection_ratio(bbox, table_bbox) < 0.2 and intersection_ratio(table_bbox, bbox) < 0.2:
            continue
        cards.append(bbox)
    if len(cards) < 2:
        return False
    # Multiple cards intersecting the candidate → card row, not a table.
    overlapping = [
        card
        for card in cards
        if intersection_area(card, table_bbox) > 0
    ]
    return len(overlapping) >= 2


def _overlaps_existing(bbox: BBox, existing: list[BBox], *, ratio: float = 0.4) -> bool:
    return any(
        intersection_ratio(bbox, other) >= ratio or intersection_ratio(other, bbox) >= ratio
        for other in existing
    )


def _grid_to_table_element(
    grid: dict[str, Any],
    *,
    text_blocks: list[TextBlock],
    page: fitz.Page,
    page_number: int,
    table_index: int,
    zebra_bands: list[_ZebraBand],
    borderless: bool,
) -> TableElement | None:
    row_count = int(grid["row_count"])
    col_count = int(grid["col_count"])
    matrix: list[list[str]] = grid["matrix"]
    bbox: BBox = grid["bbox"]
    column_widths = [float(width) for width in grid["column_widths"]]
    row_heights = [float(height) for height in grid["row_heights"]]
    zebra_fills: list[str] | None = grid.get("zebra_fills")
    if zebra_fills is None and zebra_bands:
        zebra_fills = _map_zebra_fills(bbox, row_heights, zebra_bands)

    span_index = _collect_page_spans(page)
    column_aligns = list(grid.get("column_aligns") or ["left"] * col_count)
    while len(column_aligns) < col_count:
        column_aligns.append("left")
    cells: list[TableCell] = []
    y = bbox[1]
    for row_index in range(row_count):
        x = bbox[0]
        for col_index in range(col_count):
            text = matrix[row_index][col_index] if row_index < len(matrix) else ""
            cell_bbox = (
                x,
                y,
                x + column_widths[col_index],
                y + row_heights[row_index],
            )
            cell = _build_table_cell(
                row=row_index,
                col=col_index,
                text=text,
                rowspan=1,
                colspan=1,
                cell_bbox=cell_bbox,
                text_blocks=text_blocks,
                span_index=span_index,
            )
            # Majority column alignment overrides per-cell guesses.
            cell.horizontal_align = str(column_aligns[col_index])
            if zebra_fills is not None and row_index < len(zebra_fills):
                cell.fill_color = zebra_fills[row_index]
            cells.append(cell)
            x += column_widths[col_index]
        y += row_heights[row_index]

    if not cells:
        return None

    border_color = None if borderless else "#4D4D4D"
    border_width = 0.0 if borderless else 1.0
    return TableElement(
        id=f"table_{page_number}_{table_index}",
        bbox=bbox,
        row_count=row_count,
        col_count=col_count,
        cells=cells,
        column_widths=column_widths,
        row_heights=row_heights,
        border_color=border_color,
        border_width=border_width,
        outer_border_width=border_width,
        inner_border_width=border_width,
        z_order=10_000 + table_index,
        provenance="inferred_alignment_table",
        confidence=0.75 if zebra_fills is None else 0.85,
    )


def _map_zebra_fills(
    table_bbox: BBox,
    row_heights: list[float],
    bands: list[_ZebraBand],
) -> list[str] | None:
    fills: list[str] = []
    y = table_bbox[1]
    for height in row_heights:
        row_bbox = (table_bbox[0], y, table_bbox[2], y + height)
        cy = (row_bbox[1] + row_bbox[3]) / 2.0
        matched = next(
            (
                band.fill_color
                for band in bands
                if band.bbox[1] <= cy <= band.bbox[3]
                and abs(band.bbox[0] - table_bbox[0]) <= ZEBRA_MAX_X_DRIFT_PT * 2
            ),
            None,
        )
        if matched is None:
            return None
        fills.append(matched)
        y += height
    # Need real striping, not a solid block.
    if len(set(fills)) < 2:
        return None
    return fills


__all__ = [
    "detect_inferred_tables",
]
