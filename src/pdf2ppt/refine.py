"""Deterministic render-compare-refine loop (roadmap P4).

After PPTX is built, optionally re-render each slide (LibreOffice → PDF → PNG),
compare against the source PDF page, diagnose localized problems with rule-based
checks (no LLM), apply bounded fixes to SlideIR / ``PageResult`` shapes, and
keep a change only when the page SSIM does not regress.

Opt-in via ``ConversionOptions.refine`` / CLI ``--refine``. When LibreOffice
(``soffice``) is unavailable the loop skips with a warning and never fails the
conversion.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import math
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .models import BBox, ImagePlacement, PageResult, TextBlock
from .render_compare import (
    DEFAULT_COMPARE_DPI,
    compute_pixel_error,
    compute_ssim,
    find_libreoffice_executable,
    render_pdf_page_image,
    render_pptx_pages,
    resize_to_match,
)
from .text_style import (
    choose_measurement_font,
    classify_text_script,
    measure_text_dimensions,
)

logger = logging.getLogger(__name__)

DEFAULT_REFINE_ITERS = 2
DEFAULT_MAX_FIXES_PER_SLIDE = 8
DEFAULT_FONT_SHRINK_STEP_PT = 0.5
DEFAULT_MIN_FONT_SIZE_PT = 6.0
# SSIM is insensitive to flat recolors; a modest gap still catches structural
# misses. Local MAE catches mis-coloured / missing patches that SSIM misses.
DEFAULT_LOCAL_SSIM_GAP = 0.02
DEFAULT_LOCAL_MAE_MIN = 12.0
DEFAULT_LOCAL_MAE_PAGE_RATIO = 2.5
DEFAULT_OVERFLOW_RATIO = 1.02
DEFAULT_LINE_DRIFT_PT = 3.0
DEFAULT_BBOX_EDGE_MARGIN_PT = 0.5
METRIC_EPSILON = 1e-6


@dataclass(slots=True)
class RefineIssue:
    """One localized problem found on a rendered slide."""

    kind: str
    element_id: str
    severity: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RefineFixAttempt:
    """One rule-based fix tried during refinement."""

    issue_kind: str
    element_id: str
    action: str
    accepted: bool
    metric_before: float
    metric_after: float | None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SlideRefineReport:
    """Per-slide refine summary written into the JSON report."""

    page_number: int
    iterations: int
    metric_before: float | None
    metric_after: float | None
    issues: list[RefineIssue] = field(default_factory=list)
    fixes: list[RefineFixAttempt] = field(default_factory=list)
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "iterations": self.iterations,
            "metric_before": self.metric_before,
            "metric_after": self.metric_after,
            "issues": [issue.to_dict() for issue in self.issues],
            "fixes": [fix.to_dict() for fix in self.fixes],
            "skipped_reason": self.skipped_reason,
        }


@dataclass(slots=True)
class DeckRefineReport:
    """Aggregate refine report for one conversion."""

    enabled: bool
    skipped_reason: str | None
    iterations: int
    dpi: float
    pages: list[SlideRefineReport] = field(default_factory=list)
    runtime_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "skipped_reason": self.skipped_reason,
            "iterations": self.iterations,
            "dpi": self.dpi,
            "runtime_seconds": self.runtime_seconds,
            "pages": [page.to_dict() for page in self.pages],
        }


# ---------------------------------------------------------------------------
# Geometry helpers (page points ↔ raster pixels)
# ---------------------------------------------------------------------------


def _page_to_pixel_bbox(
    bbox: BBox,
    *,
    page_width_pt: float,
    page_height_pt: float,
    image_size: tuple[int, int],
    pad_px: int = 0,
) -> tuple[int, int, int, int]:
    width_px, height_px = image_size
    scale_x = width_px / max(page_width_pt, 1.0)
    scale_y = height_px / max(page_height_pt, 1.0)
    x0 = int(math.floor(bbox[0] * scale_x)) - pad_px
    y0 = int(math.floor(bbox[1] * scale_y)) - pad_px
    x1 = int(math.ceil(bbox[2] * scale_x)) + pad_px
    y1 = int(math.ceil(bbox[3] * scale_y)) + pad_px
    return (
        max(0, x0),
        max(0, y0),
        min(width_px, x1),
        min(height_px, y1),
    )


def _union_bboxes(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _clamp_bbox_to_page(bbox: BBox, page_width_pt: float, page_height_pt: float) -> BBox:
    return (
        max(0.0, min(bbox[0], page_width_pt)),
        max(0.0, min(bbox[1], page_height_pt)),
        max(0.0, min(bbox[2], page_width_pt)),
        max(0.0, min(bbox[3], page_height_pt)),
    )


def _source_line_count(block: TextBlock) -> int:
    if block.line_bboxes:
        return max(1, len(block.line_bboxes))
    text = block.text or ""
    lines = [line for line in text.splitlines() if line.strip()]
    return max(1, len(lines) if lines else (1 if text.strip() else 0))


def _estimated_wrap_line_count(block: TextBlock) -> int:
    """How many visual lines the current font/bbox would need (measurement font)."""

    text = (block.text or "").strip()
    if not text:
        return 0
    script = classify_text_script(text)
    font_path = choose_measurement_font(script)
    if font_path is None:
        # Fall back to newline count when fonts are unavailable.
        return _source_line_count(block)
    font_size = max(DEFAULT_MIN_FONT_SIZE_PT, float(block.font_size or 12.0))
    box_width = max(1.0, _bbox_width(block.bbox))
    # Measure as a single run first; if wider than the box, approximate wrap count.
    measured_w, measured_h = measure_text_dimensions(text.replace("\n", " "), int(round(font_size)), font_path)
    if measured_w <= 0 or measured_h <= 0:
        return _source_line_count(block)
    if "\n" in text:
        # Explicit newlines already define source lines; wrap adds extras per line.
        total = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            line_w, _line_h = measure_text_dimensions(line, int(round(font_size)), font_path)
            total += max(1, int(math.ceil(line_w / box_width)))
        return max(1, total)
    return max(1, int(math.ceil(measured_w / box_width)))


def _ink_row_centers(image: Image.Image, bbox_px: tuple[int, int, int, int]) -> list[float]:
    """Y centers (in page-image pixels) of horizontal ink bands inside a crop."""

    x0, y0, x1, y1 = bbox_px
    if x1 <= x0 or y1 <= y0:
        return []
    crop = np.array(image.convert("RGB"))[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # Ink = darker than local background (presentation text on light/dark slides).
    threshold = float(np.median(gray)) - 18.0
    ink = gray < threshold
    if not np.any(ink):
        # Try inverted (light text on dark).
        ink = gray > float(np.median(gray)) + 18.0
    if not np.any(ink):
        return []
    row_density = ink.mean(axis=1)
    active = row_density > max(0.02, float(row_density.mean()) * 0.5)
    centers: list[float] = []
    start: int | None = None
    for index, is_active in enumerate(active.tolist()):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            centers.append(y0 + (start + index - 1) / 2.0)
            start = None
    if start is not None:
        centers.append(y0 + (start + len(active) - 1) / 2.0)
    return centers


def region_ssim(
    reference: Image.Image,
    candidate: Image.Image,
    bbox_px: tuple[int, int, int, int],
) -> float | None:
    """SSIM on a pixel crop; ``None`` when the crop is empty or too small."""

    x0, y0, x1, y1 = bbox_px
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    matched = resize_to_match(reference, candidate)
    ref_crop = reference.crop((x0, y0, x1, y1))
    hyp_crop = matched.crop((x0, y0, x1, y1))
    return compute_ssim(ref_crop, hyp_crop)


def region_mae(
    reference: Image.Image,
    candidate: Image.Image,
    bbox_px: tuple[int, int, int, int],
) -> float | None:
    """Mean absolute RGB error on a pixel crop; ``None`` when too small."""

    x0, y0, x1, y1 = bbox_px
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    matched = resize_to_match(reference, candidate)
    ref_crop = reference.crop((x0, y0, x1, y1))
    hyp_crop = matched.crop((x0, y0, x1, y1))
    return float(compute_pixel_error(ref_crop, hyp_crop)["pixel_mae"])


# ---------------------------------------------------------------------------
# Diagnosis (pure / deterministic; unit-tested without LibreOffice)
# ---------------------------------------------------------------------------


def diagnose_text_overflow(
    block: TextBlock,
    *,
    page_width_pt: float,
    page_height_pt: float,
    overflow_ratio: float = DEFAULT_OVERFLOW_RATIO,
) -> RefineIssue | None:
    """Flag text whose measured size exceeds its bbox or whose bbox past the slide."""

    x0, y0, x1, y1 = block.bbox
    past_edge = (
        x0 < -DEFAULT_BBOX_EDGE_MARGIN_PT
        or y0 < -DEFAULT_BBOX_EDGE_MARGIN_PT
        or x1 > page_width_pt + DEFAULT_BBOX_EDGE_MARGIN_PT
        or y1 > page_height_pt + DEFAULT_BBOX_EDGE_MARGIN_PT
    )
    text = (block.text or "").strip()
    measured_overflow = False
    measured_w = 0.0
    measured_h = 0.0
    if text:
        script = classify_text_script(text)
        font_path = choose_measurement_font(script)
        font_size = max(DEFAULT_MIN_FONT_SIZE_PT, float(block.font_size or 12.0))
        if font_path is not None:
            measured_w, measured_h = measure_text_dimensions(text, int(round(font_size)), font_path)
            box_w = max(1.0, _bbox_width(block.bbox))
            box_h = max(1.0, _bbox_height(block.bbox))
            measured_overflow = measured_w > box_w * overflow_ratio or measured_h > box_h * overflow_ratio
    if not past_edge and not measured_overflow:
        return None
    severity = 0.0
    if past_edge:
        severity += 0.5
    if measured_overflow:
        box_w = max(1.0, _bbox_width(block.bbox))
        box_h = max(1.0, _bbox_height(block.bbox))
        severity += min(1.0, max(measured_w / box_w, measured_h / box_h, 1.0) - 1.0)
    return RefineIssue(
        kind="text_overflow",
        element_id=block.id,
        severity=round(severity, 4),
        details={
            "past_slide_edge": past_edge,
            "measured_width_pt": round(measured_w, 3),
            "measured_height_pt": round(measured_h, 3),
            "bbox": list(block.bbox),
            "font_size": block.font_size,
        },
    )


def diagnose_line_drift(
    block: TextBlock,
    *,
    rendered_line_ys_pt: list[float],
    drift_pt: float = DEFAULT_LINE_DRIFT_PT,
) -> RefineIssue | None:
    """Flag when rendered line y-centers drift from PDF ``line_bboxes``."""

    if not block.line_bboxes or not rendered_line_ys_pt:
        return None
    expected = sorted((box[1] + box[3]) / 2.0 for box in block.line_bboxes)
    observed = sorted(rendered_line_ys_pt)
    pair_count = min(len(expected), len(observed))
    if pair_count == 0:
        return None
    drifts = [abs(expected[index] - observed[index]) for index in range(pair_count)]
    max_drift = max(drifts)
    if max_drift <= drift_pt:
        return None
    return RefineIssue(
        kind="line_drift",
        element_id=block.id,
        severity=round(min(1.0, max_drift / max(drift_pt * 4.0, 1.0)), 4),
        details={
            "max_drift_pt": round(max_drift, 3),
            "expected_ys_pt": [round(value, 3) for value in expected],
            "observed_ys_pt": [round(value, 3) for value in observed],
        },
    )


def diagnose_extra_wrap(block: TextBlock) -> RefineIssue | None:
    """Flag text that would wrap onto more lines than the source."""

    source_lines = _source_line_count(block)
    if source_lines <= 0:
        return None
    estimated = _estimated_wrap_line_count(block)
    if estimated <= source_lines:
        return None
    return RefineIssue(
        kind="extra_wrap",
        element_id=block.id,
        severity=round(min(1.0, (estimated - source_lines) / max(source_lines, 1)), 4),
        details={
            "source_line_count": source_lines,
            "estimated_line_count": estimated,
            "font_size": block.font_size,
            "bbox": list(block.bbox),
        },
    )


def diagnose_low_local_ssim(
    *,
    element_id: str,
    local_ssim: float | None,
    page_ssim: float,
    gap: float = DEFAULT_LOCAL_SSIM_GAP,
    local_mae: float | None = None,
    page_mae: float = 0.0,
    mae_min: float = DEFAULT_LOCAL_MAE_MIN,
    mae_page_ratio: float = DEFAULT_LOCAL_MAE_PAGE_RATIO,
) -> RefineIssue | None:
    """Flag an element region that is visually far worse than the page average.

    Triggers when local SSIM is below ``page_ssim - gap``, or when local MAE is
    both above ``mae_min`` and clearly worse than the page MAE (flat recolors
    often keep high SSIM).
    """

    ssim_bad = local_ssim is not None and local_ssim < page_ssim - gap
    mae_bad = local_mae is not None and local_mae >= mae_min and local_mae >= max(mae_min, page_mae * mae_page_ratio)
    if not ssim_bad and not mae_bad:
        return None
    severity = 0.0
    if ssim_bad and local_ssim is not None:
        severity = max(severity, min(1.0, (page_ssim - gap) - local_ssim))
    if mae_bad and local_mae is not None:
        severity = max(severity, min(1.0, local_mae / 64.0))
    return RefineIssue(
        kind="low_local_ssim",
        element_id=element_id,
        severity=round(severity, 4),
        details={
            "local_ssim": None if local_ssim is None else round(local_ssim, 6),
            "page_ssim": round(page_ssim, 6),
            "gap": gap,
            "local_mae": None if local_mae is None else round(local_mae, 4),
            "page_mae": round(page_mae, 4),
            "ssim_bad": ssim_bad,
            "mae_bad": mae_bad,
        },
    )


def diagnose_slide(
    reference: Image.Image,
    candidate: Image.Image,
    page_result: PageResult,
    *,
    page_ssim: float | None = None,
    local_ssim_gap: float = DEFAULT_LOCAL_SSIM_GAP,
) -> tuple[float, list[RefineIssue]]:
    """Run all deterministic per-slide checks. Returns ``(page_ssim, issues)``."""

    matched = resize_to_match(reference, candidate)
    ssim = page_ssim if page_ssim is not None else compute_ssim(reference, matched)
    page_mae = float(compute_pixel_error(reference, matched)["pixel_mae"])
    issues: list[RefineIssue] = []

    def _check_region(element_id: str, bbox_px: tuple[int, int, int, int]) -> None:
        local = region_ssim(reference, matched, bbox_px)
        local_mae = region_mae(reference, matched, bbox_px)
        low = diagnose_low_local_ssim(
            element_id=element_id,
            local_ssim=local,
            page_ssim=ssim,
            gap=local_ssim_gap,
            local_mae=local_mae,
            page_mae=page_mae,
        )
        if low is not None:
            issues.append(low)

    for block in page_result.text_blocks:
        overflow = diagnose_text_overflow(
            block,
            page_width_pt=page_result.width_pt,
            page_height_pt=page_result.height_pt,
        )
        if overflow is not None:
            issues.append(overflow)
        wrap = diagnose_extra_wrap(block)
        if wrap is not None:
            issues.append(wrap)

        bbox_px = _page_to_pixel_bbox(
            block.bbox,
            page_width_pt=page_result.width_pt,
            page_height_pt=page_result.height_pt,
            image_size=reference.size,
            pad_px=4,
        )
        if block.line_bboxes:
            ink_ys_px = _ink_row_centers(matched, bbox_px)
            scale_y = reference.size[1] / max(page_result.height_pt, 1.0)
            ink_ys_pt = [y / scale_y for y in ink_ys_px]
            drift = diagnose_line_drift(block, rendered_line_ys_pt=ink_ys_pt)
            if drift is not None:
                issues.append(drift)

        _check_region(block.id, bbox_px)

    for shape in page_result.shape_elements:
        bbox_px = _page_to_pixel_bbox(
            shape.bbox,
            page_width_pt=page_result.width_pt,
            page_height_pt=page_result.height_pt,
            image_size=reference.size,
            pad_px=2,
        )
        _check_region(shape.id, bbox_px)

    for index, image_element in enumerate(page_result.image_elements):
        element_id = f"image_{index}"
        bbox_px = _page_to_pixel_bbox(
            image_element.bbox,
            page_width_pt=page_result.width_pt,
            page_height_pt=page_result.height_pt,
            image_size=reference.size,
            pad_px=2,
        )
        _check_region(element_id, bbox_px)

    issues.sort(key=lambda item: (-item.severity, item.kind, item.element_id))
    return ssim, issues


# ---------------------------------------------------------------------------
# Rule-based fixes (mutate a deep-copied PageResult; recorded for the report)
# ---------------------------------------------------------------------------


def _find_text_block(page: PageResult, element_id: str) -> TextBlock | None:
    for block in page.text_blocks:
        if block.id == element_id:
            return block
    return None


def _replace_text_block(page: PageResult, updated: TextBlock) -> None:
    page.text_blocks = [updated if block.id == updated.id else block for block in page.text_blocks]


def apply_shrink_font(
    page: PageResult, element_id: str, *, step_pt: float = DEFAULT_FONT_SHRINK_STEP_PT
) -> dict[str, Any] | None:
    block = _find_text_block(page, element_id)
    if block is None:
        return None
    current = float(block.font_size or 12.0)
    new_size = max(DEFAULT_MIN_FONT_SIZE_PT, round(current - step_pt, 2))
    if new_size >= current - 1e-9:
        return None
    _replace_text_block(page, replace(block, font_size=new_size))
    return {"font_size_before": current, "font_size_after": new_size}


def apply_match_line_bboxes(page: PageResult, element_id: str) -> dict[str, Any] | None:
    """Widen / reposition a text box to the union of its source ``line_bboxes``."""

    block = _find_text_block(page, element_id)
    if block is None or not block.line_bboxes:
        return None
    union = _union_bboxes(list(block.line_bboxes))
    if union is None:
        return None
    # Small padding so glyphs are not clipped at the box edge.
    pad = max(0.5, (union[3] - union[1]) * 0.05)
    new_bbox = _clamp_bbox_to_page(
        (union[0] - pad, union[1] - pad, union[2] + pad, union[3] + pad),
        page.width_pt,
        page.height_pt,
    )
    if new_bbox == block.bbox:
        return None
    before = list(block.bbox)
    _replace_text_block(page, replace(block, bbox=new_bbox))
    return {"bbox_before": before, "bbox_after": list(new_bbox)}


def apply_widen_for_wrap(page: PageResult, element_id: str) -> dict[str, Any] | None:
    """Widen a text box so the measured single-line width fits (no extra wrap)."""

    block = _find_text_block(page, element_id)
    if block is None:
        return None
    text = (block.text or "").strip()
    if not text:
        return None
    script = classify_text_script(text)
    font_path = choose_measurement_font(script)
    if font_path is None:
        return None
    font_size = max(DEFAULT_MIN_FONT_SIZE_PT, float(block.font_size or 12.0))
    source_lines = max(1, _source_line_count(block))
    # Target width: enough for the longest source line (or whole text if one line).
    lines = [line for line in text.splitlines() if line.strip()] or [text.replace("\n", " ")]
    max_line_w = 0.0
    for line in lines:
        line_w, _line_h = measure_text_dimensions(line, int(round(font_size)), font_path)
        max_line_w = max(max_line_w, line_w)
    # If source has N lines, allow height for N; widen only for horizontal overflow.
    needed_width = max_line_w * 1.02
    current_width = _bbox_width(block.bbox)
    if needed_width <= current_width:
        return None
    x0, y0, x1, y1 = block.bbox
    # Prefer expanding to the right, then left, clamped to the page.
    extra = needed_width - current_width
    new_x1 = min(page.width_pt, x1 + extra)
    remaining = needed_width - (new_x1 - x0)
    new_x0 = max(0.0, x0 - max(0.0, remaining))
    # Grow height slightly if source is multi-line and box is too short.
    line_h = max(font_size * 1.2, _bbox_height(block.bbox) / source_lines)
    needed_height = line_h * source_lines
    new_y1 = min(page.height_pt, max(y1, y0 + needed_height))
    new_bbox = (new_x0, y0, new_x1, new_y1)
    if new_bbox == block.bbox:
        return None
    before = list(block.bbox)
    _replace_text_block(page, replace(block, bbox=new_bbox))
    return {"bbox_before": before, "bbox_after": list(new_bbox), "needed_width_pt": round(needed_width, 3)}


def apply_raster_fallback(
    page: PageResult,
    element_id: str,
    *,
    reference_page_image: Image.Image,
) -> dict[str, Any] | None:
    """Last resort: paste a PDF-page crop for a badly wrong element region.

    Text stays editable: the crop is added as an ``ImagePlacement`` behind the
    existing text box (lower z-order). Shape elements that match ``element_id``
    are removed so the raster is not double-covered by a wrong fill.
    """

    bbox: BBox | None = None
    kind = "unknown"
    block = _find_text_block(page, element_id)
    if block is not None:
        bbox = block.bbox
        kind = "text"
    else:
        for shape in page.shape_elements:
            if shape.id == element_id:
                bbox = shape.bbox
                kind = "shape"
                break
        if bbox is None and element_id.startswith("image_"):
            try:
                index = int(element_id.split("_", 1)[1])
            except (IndexError, ValueError):
                index = -1
            if 0 <= index < len(page.image_elements):
                bbox = page.image_elements[index].bbox
                kind = "image"

    if bbox is None:
        return None

    bbox_px = _page_to_pixel_bbox(
        bbox,
        page_width_pt=page.width_pt,
        page_height_pt=page.height_pt,
        image_size=reference_page_image.size,
        pad_px=1,
    )
    x0, y0, x1, y1 = bbox_px
    if x1 <= x0 or y1 <= y0:
        return None
    crop = reference_page_image.crop((x0, y0, x1, y1)).convert("RGB")
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    # Place under typical text z-order so text remains editable on top.
    placement = ImagePlacement(bbox=bbox, png_bytes=png_bytes, z_order=-10)
    page.image_elements = list(page.image_elements) + [placement]

    removed_shape = False
    if kind == "shape":
        before_count = len(page.shape_elements)
        page.shape_elements = [shape for shape in page.shape_elements if shape.id != element_id]
        removed_shape = len(page.shape_elements) < before_count

    return {
        "kind": kind,
        "bbox": list(bbox),
        "crop_px": [x0, y0, x1, y1],
        "removed_shape": removed_shape,
        "bytes": len(png_bytes),
    }


def propose_fixes_for_issue(issue: RefineIssue) -> list[str]:
    """Ordered fix actions for an issue kind (deterministic priority)."""

    if issue.kind in {"text_overflow", "extra_wrap"}:
        return ["shrink_font", "widen_bbox", "match_line_bboxes", "raster_fallback"]
    if issue.kind == "line_drift":
        return ["match_line_bboxes", "widen_bbox", "shrink_font", "raster_fallback"]
    if issue.kind == "low_local_ssim":
        return ["match_line_bboxes", "shrink_font", "raster_fallback"]
    return ["raster_fallback"]


def apply_fix_action(
    page: PageResult,
    *,
    element_id: str,
    action: str,
    reference_page_image: Image.Image | None,
) -> dict[str, Any] | None:
    if action == "shrink_font":
        return apply_shrink_font(page, element_id)
    if action == "widen_bbox":
        return apply_widen_for_wrap(page, element_id)
    if action == "match_line_bboxes":
        return apply_match_line_bboxes(page, element_id)
    if action == "raster_fallback":
        if reference_page_image is None:
            return None
        return apply_raster_fallback(page, element_id, reference_page_image=reference_page_image)
    return None


# ---------------------------------------------------------------------------
# Accept / reject loop
# ---------------------------------------------------------------------------


RenderPagesFn = Callable[..., list[Image.Image]]
RenderPdfPageFn = Callable[..., Image.Image]
WritePptxFn = Callable[..., Path]


def _deepcopy_pages(pages: list[PageResult]) -> list[PageResult]:
    return copy.deepcopy(pages)


def refine_pages(
    pages: list[PageResult],
    *,
    pdf_path: Path,
    output_pptx: Path,
    slide_width_pt: float,
    slide_height_pt: float,
    iterations: int = DEFAULT_REFINE_ITERS,
    max_fixes_per_slide: int = DEFAULT_MAX_FIXES_PER_SLIDE,
    dpi: float = DEFAULT_COMPARE_DPI,
    write_pptx: WritePptxFn | None = None,
    render_pptx: RenderPagesFn | None = None,
    render_pdf_page: RenderPdfPageFn | None = None,
    work_dir: Path | None = None,
) -> tuple[list[PageResult], DeckRefineReport]:
    """Run the render-compare-refine loop. Mutates nothing until a fix is accepted.

    Returns the (possibly improved) page list and a deck-level report. When
    LibreOffice is missing, returns the original pages and a skipped report.
    """

    from time import perf_counter

    from .pipeline import render_pages_to_pptx

    started = perf_counter()
    iters = max(0, int(iterations))
    if iters <= 0:
        return pages, DeckRefineReport(
            enabled=True,
            skipped_reason="refine_iters<=0",
            iterations=0,
            dpi=dpi,
            runtime_seconds=0.0,
        )

    if find_libreoffice_executable() is None and render_pptx is None:
        logger.warning(
            "Refine skipped: LibreOffice (soffice) is not available. Install libreoffice-impress to enable --refine."
        )
        return pages, DeckRefineReport(
            enabled=True,
            skipped_reason="libreoffice_unavailable",
            iterations=iters,
            dpi=dpi,
            runtime_seconds=round(perf_counter() - started, 3),
        )

    write_fn = write_pptx or render_pages_to_pptx
    render_pptx_fn = render_pptx or render_pptx_pages
    render_pdf_fn = render_pdf_page or render_pdf_page_image

    working = _deepcopy_pages(pages)
    owns_work = work_dir is None
    root = Path(tempfile.mkdtemp(prefix="pdf2ppt-refine-")) if owns_work else Path(work_dir or ".")
    root.mkdir(parents=True, exist_ok=True)

    try:
        # Baseline render.
        write_fn(
            working,
            output_pptx,
            slide_width_pt=slide_width_pt,
            slide_height_pt=slide_height_pt,
        )
        rendered = render_pptx_fn(output_pptx, dpi=dpi, work_dir=root / "render0")
        page_reports: list[SlideRefineReport] = []

        for page_index, page in enumerate(working):
            if page_index >= len(rendered):
                page_reports.append(
                    SlideRefineReport(
                        page_number=page.page_number,
                        iterations=0,
                        metric_before=None,
                        metric_after=None,
                        skipped_reason="missing_rendered_slide",
                    )
                )
                continue

            reference = render_pdf_fn(pdf_path, page_index, dpi=dpi)
            candidate = rendered[page_index]
            metric_before, issues = diagnose_slide(reference, candidate, page)
            slide_report = SlideRefineReport(
                page_number=page.page_number,
                iterations=0,
                metric_before=metric_before,
                metric_after=metric_before,
                issues=list(issues),
            )
            current_metric = metric_before
            fixes_applied = 0

            for iteration in range(iters):
                if fixes_applied >= max_fixes_per_slide:
                    break
                # Re-diagnose on the latest candidate each iteration.
                _, issues = diagnose_slide(reference, candidate, page, page_ssim=current_metric)
                if not issues:
                    break
                slide_report.iterations = iteration + 1
                made_progress = False

                for issue in issues:
                    if fixes_applied >= max_fixes_per_slide:
                        break
                    for action in propose_fixes_for_issue(issue):
                        if fixes_applied >= max_fixes_per_slide:
                            break
                        snapshot = _deepcopy_pages(working)
                        trial_page = working[page_index]
                        details = apply_fix_action(
                            trial_page,
                            element_id=issue.element_id,
                            action=action,
                            reference_page_image=reference,
                        )
                        if details is None:
                            continue

                        # Re-render full deck (LibreOffice works on whole PPTX).
                        write_fn(
                            working,
                            output_pptx,
                            slide_width_pt=slide_width_pt,
                            slide_height_pt=slide_height_pt,
                        )
                        trial_rendered = render_pptx_fn(
                            output_pptx,
                            dpi=dpi,
                            work_dir=root / f"render-p{page_index}-i{iteration}-{action}",
                        )
                        if page_index >= len(trial_rendered):
                            working[:] = snapshot
                            page = working[page_index]
                            write_fn(
                                working,
                                output_pptx,
                                slide_width_pt=slide_width_pt,
                                slide_height_pt=slide_height_pt,
                            )
                            slide_report.fixes.append(
                                RefineFixAttempt(
                                    issue_kind=issue.kind,
                                    element_id=issue.element_id,
                                    action=action,
                                    accepted=False,
                                    metric_before=current_metric,
                                    metric_after=None,
                                    details={**details, "reject_reason": "missing_rendered_slide"},
                                )
                            )
                            continue

                        trial_candidate = trial_rendered[page_index]
                        trial_metric = compute_ssim(reference, resize_to_match(reference, trial_candidate))
                        accepted = trial_metric + METRIC_EPSILON >= current_metric
                        slide_report.fixes.append(
                            RefineFixAttempt(
                                issue_kind=issue.kind,
                                element_id=issue.element_id,
                                action=action,
                                accepted=accepted,
                                metric_before=current_metric,
                                metric_after=trial_metric,
                                details=details,
                            )
                        )
                        if accepted:
                            current_metric = trial_metric
                            candidate = trial_candidate
                            rendered = trial_rendered
                            page = working[page_index]
                            fixes_applied += 1
                            made_progress = True
                            # One accepted fix per issue, then re-diagnose.
                            break
                        # Revert and restore PPTX on disk.
                        working[:] = snapshot
                        page = working[page_index]
                        write_fn(
                            working,
                            output_pptx,
                            slide_width_pt=slide_width_pt,
                            slide_height_pt=slide_height_pt,
                        )
                    else:
                        continue
                    break  # next iteration after one accepted issue fix

                if not made_progress:
                    break

            slide_report.metric_after = current_metric
            page_reports.append(slide_report)

        # Final write of accepted pages.
        write_fn(
            working,
            output_pptx,
            slide_width_pt=slide_width_pt,
            slide_height_pt=slide_height_pt,
        )
        return working, DeckRefineReport(
            enabled=True,
            skipped_reason=None,
            iterations=iters,
            dpi=dpi,
            pages=page_reports,
            runtime_seconds=round(perf_counter() - started, 3),
        )
    finally:
        if owns_work:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


def write_refine_report(report: DeckRefineReport, path: Path) -> Path:
    """Write the per-slide refine JSON report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


__all__ = [
    "DEFAULT_MAX_FIXES_PER_SLIDE",
    "DEFAULT_REFINE_ITERS",
    "DeckRefineReport",
    "RefineFixAttempt",
    "RefineIssue",
    "SlideRefineReport",
    "apply_fix_action",
    "apply_match_line_bboxes",
    "apply_raster_fallback",
    "apply_shrink_font",
    "apply_widen_for_wrap",
    "diagnose_extra_wrap",
    "diagnose_line_drift",
    "diagnose_low_local_ssim",
    "diagnose_slide",
    "diagnose_text_overflow",
    "propose_fixes_for_issue",
    "refine_pages",
    "region_ssim",
    "write_refine_report",
]
