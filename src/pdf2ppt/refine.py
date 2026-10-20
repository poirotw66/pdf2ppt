"""Deterministic render-compare-refine loop (roadmap P4).

After PPTX is built, optionally re-render each slide (LibreOffice → PDF → PNG),
compare against the source PDF page, diagnose localized problems with rule-based
checks (no LLM), apply bounded fixes to SlideIR / ``PageResult`` shapes, and
keep a change only when page SSIM **strictly improves** and editability / text
content do not regress.

Opt-in via ``ConversionOptions.refine`` / CLI ``--refine``. When LibreOffice
(``soffice``) is unavailable **or fails**, the loop skips with a warning,
restores the last accepted PPTX, and never fails the conversion.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import math
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation

from .models import BBox, ImagePlacement, PageResult, TextBlock, TextParagraph
from .render_compare import (
    DEFAULT_COMPARE_DPI,
    RenderCompareError,
    compute_pixel_error,
    compute_ssim,
    extract_pdf_page_text,
    find_libreoffice_executable,
    inspect_slide_editability,
    render_pdf_page_image,
    render_pptx_pages,
    resize_to_match,
    text_token_f1,
)
from .text_style import (
    choose_measurement_font,
    classify_text_script,
    measure_text_dimensions,
)

logger = logging.getLogger(__name__)

DEFAULT_REFINE_ITERS = 2
DEFAULT_MAX_FIXES_PER_SLIDE = 8
DEFAULT_MAX_ATTEMPTS_PER_SLIDE = 24
DEFAULT_TIME_BUDGET_SECONDS = 90.0
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
# Strict improvement required (neutral / flat accepts are rejected).
METRIC_IMPROVE_EPSILON = 1e-4
EDITABILITY_EPSILON = 1e-6
TEXT_F1_EPSILON = 1e-6


@dataclass(slots=True)
class RefineIssue:
    """One localized problem found on a rendered slide."""

    kind: str
    element_id: str
    severity: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def has_render_evidence(self) -> bool:
        return bool(self.details.get("render_evidence"))

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
    xml_changed: bool | None = None
    noop: bool = False

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
    attempt_count: int = 0
    soffice_call_count: int = 0
    editability_before: float | None = None
    editability_after: float | None = None
    text_f1_before: float | None = None
    text_f1_after: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "iterations": self.iterations,
            "metric_before": self.metric_before,
            "metric_after": self.metric_after,
            "issues": [issue.to_dict() for issue in self.issues],
            "fixes": [fix.to_dict() for fix in self.fixes],
            "skipped_reason": self.skipped_reason,
            "attempt_count": self.attempt_count,
            "soffice_call_count": self.soffice_call_count,
            "editability_before": self.editability_before,
            "editability_after": self.editability_after,
            "text_f1_before": self.text_f1_before,
            "text_f1_after": self.text_f1_after,
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
    soffice_call_count: int = 0
    attempt_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "skipped_reason": self.skipped_reason,
            "iterations": self.iterations,
            "dpi": self.dpi,
            "runtime_seconds": self.runtime_seconds,
            "soffice_call_count": self.soffice_call_count,
            "attempt_count": self.attempt_count,
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


def _bboxes_intersect(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


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


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    threshold = float(np.median(gray)) - 18.0
    ink = gray < threshold
    if not np.any(ink):
        ink = gray > float(np.median(gray)) + 18.0
    return ink


def _ink_row_centers(image: Image.Image, bbox_px: tuple[int, int, int, int]) -> list[float]:
    """Y centers (in page-image pixels) of horizontal ink bands inside a crop."""

    x0, y0, x1, y1 = bbox_px
    if x1 <= x0 or y1 <= y0:
        return []
    crop = np.array(image.convert("RGB"))[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    ink = _ink_mask(gray)
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


def _ink_extent_px(image: Image.Image, bbox_px: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Bounding box of ink pixels inside ``bbox_px``, or ``None`` if empty."""

    x0, y0, x1, y1 = bbox_px
    if x1 <= x0 or y1 <= y0:
        return None
    crop = np.array(image.convert("RGB"))[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    ink = _ink_mask(gray)
    if not np.any(ink):
        return None
    ys, xs = np.where(ink)
    return (
        int(x0 + xs.min()),
        int(y0 + ys.min()),
        int(x0 + xs.max() + 1),
        int(y0 + ys.max() + 1),
    )


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
# PPTX helpers (XML digest, text extract, editability, single-slide write)
# ---------------------------------------------------------------------------


def pptx_content_digest(pptx_path: Path) -> str:
    """Stable hash of slide XML + media so no-op writes are detectable."""

    digest = hashlib.sha256()
    with zipfile.ZipFile(pptx_path) as archive:
        names = sorted(
            name for name in archive.namelist() if name.startswith("ppt/slides/") or name.startswith("ppt/media/")
        )
        for name in names:
            digest.update(name.encode("utf-8"))
            digest.update(archive.read(name))
    return digest.hexdigest()


def extract_pptx_slide_text(pptx_path: Path, *, slide_index: int = 0) -> str:
    """Concatenate editable text-frame text from one slide (no OCR)."""

    presentation = Presentation(str(pptx_path))
    if slide_index < 0 or slide_index >= len(presentation.slides):
        return ""
    slide = presentation.slides[slide_index]
    parts: list[str] = []
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
            text = (shape.text_frame.text or "").strip()
            if text:
                parts.append(text)
        if getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    cell_text = (cell.text or "").strip()
                    if cell_text:
                        parts.append(cell_text)
    return "\n".join(parts)


def slide_editability_score(pptx_path: Path, *, slide_index: int = 0) -> float:
    presentation = Presentation(str(pptx_path))
    if slide_index < 0 or slide_index >= len(presentation.slides):
        return 0.0
    metrics = inspect_slide_editability(
        presentation.slides[slide_index],
        slide_width=int(presentation.slide_width or 1),
        slide_height=int(presentation.slide_height or 1),
    )
    return float(metrics.score)


def slide_text_f1_vs_pdf(pptx_path: Path, pdf_path: Path, *, page_index: int) -> float:
    """Bag-of-tokens F1: PDF page text vs editable PPTX text (no OCR)."""

    reference = extract_pdf_page_text(pdf_path, page_index)
    presentation = Presentation(str(pptx_path))
    # Single-slide trial PPTX always has one slide at index 0.
    slide_index = 0 if len(presentation.slides) == 1 else page_index
    candidate = extract_pptx_slide_text(pptx_path, slide_index=slide_index)
    return float(text_token_f1(reference, candidate)["text_f1"])


# ---------------------------------------------------------------------------
# Diagnosis (pure / deterministic; unit-tested without LibreOffice)
# ---------------------------------------------------------------------------


def diagnose_text_overflow(
    block: TextBlock,
    *,
    page_width_pt: float,
    page_height_pt: float,
    candidate: Image.Image | None = None,
    overflow_ratio: float = DEFAULT_OVERFLOW_RATIO,
) -> RefineIssue | None:
    """Flag overflow from render ink spill, or (lower priority) measurement."""

    x0, y0, x1, y1 = block.bbox
    past_edge = (
        x0 < -DEFAULT_BBOX_EDGE_MARGIN_PT
        or y0 < -DEFAULT_BBOX_EDGE_MARGIN_PT
        or x1 > page_width_pt + DEFAULT_BBOX_EDGE_MARGIN_PT
        or y1 > page_height_pt + DEFAULT_BBOX_EDGE_MARGIN_PT
    )

    render_spill = False
    spill_px = 0
    if candidate is not None:
        # Look slightly outside the box for ink that belongs to this block.
        box_px = _page_to_pixel_bbox(
            block.bbox,
            page_width_pt=page_width_pt,
            page_height_pt=page_height_pt,
            image_size=candidate.size,
            pad_px=0,
        )
        search_px = _page_to_pixel_bbox(
            block.bbox,
            page_width_pt=page_width_pt,
            page_height_pt=page_height_pt,
            image_size=candidate.size,
            pad_px=8,
        )
        extent = _ink_extent_px(candidate, search_px)
        if extent is not None:
            spill_left = max(0, box_px[0] - extent[0])
            spill_top = max(0, box_px[1] - extent[1])
            spill_right = max(0, extent[2] - box_px[2])
            spill_bottom = max(0, extent[3] - box_px[3])
            spill_px = spill_left + spill_top + spill_right + spill_bottom
            # Also treat ink past the slide edge as overflow.
            edge_spill = (
                extent[0] <= 1
                or extent[1] <= 1
                or extent[2] >= candidate.size[0] - 1
                or extent[3] >= candidate.size[1] - 1
            )
            render_spill = spill_px >= 3 or (edge_spill and past_edge)

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

    if not past_edge and not render_spill and not measured_overflow:
        return None

    # Prefer render evidence; measurement-only is demoted so it does not eat budget.
    render_evidence = render_spill or past_edge
    if not render_evidence and measured_overflow:
        severity = 0.15
    else:
        severity = 0.5 if past_edge else 0.0
        if render_spill:
            severity = max(severity, min(1.0, spill_px / 20.0))
        if measured_overflow and render_evidence:
            box_w = max(1.0, _bbox_width(block.bbox))
            box_h = max(1.0, _bbox_height(block.bbox))
            severity = max(
                severity,
                min(1.0, max(measured_w / box_w, measured_h / box_h, 1.0) - 1.0),
            )

    return RefineIssue(
        kind="text_overflow",
        element_id=block.id,
        severity=round(severity, 4),
        details={
            "past_slide_edge": past_edge,
            "render_spill": render_spill,
            "spill_px": spill_px,
            "measured_width_pt": round(measured_w, 3),
            "measured_height_pt": round(measured_h, 3),
            "bbox": list(block.bbox),
            "font_size": block.font_size,
            "render_evidence": render_evidence,
            "element_kind": "text",
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
            "render_evidence": True,
            "element_kind": "text",
        },
    )


def diagnose_extra_wrap(
    block: TextBlock,
    *,
    candidate: Image.Image | None = None,
    page_width_pt: float | None = None,
    page_height_pt: float | None = None,
) -> RefineIssue | None:
    """Flag extra wrapping only when render ink shows more lines than the source."""

    source_lines = _source_line_count(block)
    if source_lines <= 0:
        return None
    if candidate is None or page_width_pt is None or page_height_pt is None:
        return None

    bbox_px = _page_to_pixel_bbox(
        block.bbox,
        page_width_pt=page_width_pt,
        page_height_pt=page_height_pt,
        image_size=candidate.size,
        pad_px=4,
    )
    ink_rows = _ink_row_centers(candidate, bbox_px)
    rendered_lines = len(ink_rows)
    if rendered_lines <= source_lines:
        return None
    return RefineIssue(
        kind="extra_wrap",
        element_id=block.id,
        severity=round(min(1.0, (rendered_lines - source_lines) / max(source_lines, 1)), 4),
        details={
            "source_line_count": source_lines,
            "estimated_line_count": rendered_lines,
            "font_size": block.font_size,
            "bbox": list(block.bbox),
            "render_evidence": True,
            "element_kind": "text",
        },
    )


def diagnose_low_local_ssim(
    *,
    element_id: str,
    element_kind: str,
    local_ssim: float | None,
    page_ssim: float,
    gap: float = DEFAULT_LOCAL_SSIM_GAP,
    local_mae: float | None = None,
    page_mae: float = 0.0,
    mae_min: float = DEFAULT_LOCAL_MAE_MIN,
    mae_page_ratio: float = DEFAULT_LOCAL_MAE_PAGE_RATIO,
) -> RefineIssue | None:
    """Flag an element region that is visually far worse than the page average."""

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
            "render_evidence": True,
            "element_kind": element_kind,
        },
    )


def _issue_sort_key(issue: RefineIssue) -> tuple[int, float, str, str]:
    # Render-evidence issues first, then higher severity.
    return (0 if issue.has_render_evidence else 1, -issue.severity, issue.kind, issue.element_id)


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

    def _check_region(element_id: str, element_kind: str, bbox_px: tuple[int, int, int, int]) -> None:
        local = region_ssim(reference, matched, bbox_px)
        local_mae = region_mae(reference, matched, bbox_px)
        low = diagnose_low_local_ssim(
            element_id=element_id,
            element_kind=element_kind,
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
            candidate=matched,
        )
        if overflow is not None:
            issues.append(overflow)
        wrap = diagnose_extra_wrap(
            block,
            candidate=matched,
            page_width_pt=page_result.width_pt,
            page_height_pt=page_result.height_pt,
        )
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

        _check_region(block.id, "text", bbox_px)

    for shape in page_result.shape_elements:
        bbox_px = _page_to_pixel_bbox(
            shape.bbox,
            page_width_pt=page_result.width_pt,
            page_height_pt=page_result.height_pt,
            image_size=reference.size,
            pad_px=2,
        )
        _check_region(shape.id, "shape", bbox_px)

    for index, image_element in enumerate(page_result.image_elements):
        element_id = f"image_{index}"
        bbox_px = _page_to_pixel_bbox(
            image_element.bbox,
            page_width_pt=page_result.width_pt,
            page_height_pt=page_result.height_pt,
            image_size=reference.size,
            pad_px=2,
        )
        _check_region(element_id, "image", bbox_px)

    issues.sort(key=_issue_sort_key)
    return ssim, issues


# ---------------------------------------------------------------------------
# Rule-based fixes
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
    """Shrink ``block.font_size`` and every paragraph ``font_size`` (writer prefers paras)."""

    block = _find_text_block(page, element_id)
    if block is None:
        return None
    current = float(block.font_size or 12.0)
    new_size = max(DEFAULT_MIN_FONT_SIZE_PT, round(current - step_pt, 2))
    paragraphs = list(block.paragraphs or [])
    para_changed = False
    new_paragraphs: list[TextParagraph] | None = None
    if paragraphs:
        new_paragraphs = []
        for para in paragraphs:
            para_size = float(para.font_size) if para.font_size is not None else current
            para_new = max(DEFAULT_MIN_FONT_SIZE_PT, round(para_size - step_pt, 2))
            if para_new < para_size - 1e-9:
                para_changed = True
            new_paragraphs.append(replace(para, font_size=para_new))
    if new_size >= current - 1e-9 and not para_changed:
        return None
    updated = replace(
        block,
        font_size=new_size if new_size < current - 1e-9 else block.font_size,
        paragraphs=new_paragraphs if new_paragraphs is not None else block.paragraphs,
    )
    _replace_text_block(page, updated)
    return {
        "font_size_before": current,
        "font_size_after": updated.font_size,
        "paragraph_sizes_scaled": para_changed,
    }


def apply_match_line_bboxes(page: PageResult, element_id: str) -> dict[str, Any] | None:
    """Widen / reposition a text box to the union of its source ``line_bboxes``."""

    block = _find_text_block(page, element_id)
    if block is None or not block.line_bboxes:
        return None
    union = _union_bboxes(list(block.line_bboxes))
    if union is None:
        return None
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
    lines = [line for line in text.splitlines() if line.strip()] or [text.replace("\n", " ")]
    max_line_w = 0.0
    for line in lines:
        line_w, _line_h = measure_text_dimensions(line, int(round(font_size)), font_path)
        max_line_w = max(max_line_w, line_w)
    needed_width = max_line_w * 1.02
    current_width = _bbox_width(block.bbox)
    if needed_width <= current_width:
        return None
    x0, y0, x1, y1 = block.bbox
    extra = needed_width - current_width
    new_x1 = min(page.width_pt, x1 + extra)
    remaining = needed_width - (new_x1 - x0)
    new_x0 = max(0.0, x0 - max(0.0, remaining))
    line_h = max(font_size * 1.2, _bbox_height(block.bbox) / source_lines)
    needed_height = line_h * source_lines
    new_y1 = min(page.height_pt, max(y1, y0 + needed_height))
    new_bbox = (new_x0, y0, new_x1, new_y1)
    if new_bbox == block.bbox:
        return None
    before = list(block.bbox)
    _replace_text_block(page, replace(block, bbox=new_bbox))
    return {"bbox_before": before, "bbox_after": list(new_bbox), "needed_width_pt": round(needed_width, 3)}


def _erase_overlapping_text_from_crop(
    crop: Image.Image,
    *,
    crop_bbox_px: tuple[int, int, int, int],
    page: PageResult,
    image_size: tuple[int, int],
) -> Image.Image:
    """Paint over glyphs of editable text that intersect a non-text raster crop."""

    cx0, cy0, cx1, cy1 = crop_bbox_px
    crop_box_pt = (
        cx0 * page.width_pt / max(image_size[0], 1),
        cy0 * page.height_pt / max(image_size[1], 1),
        cx1 * page.width_pt / max(image_size[0], 1),
        cy1 * page.height_pt / max(image_size[1], 1),
    )
    result = crop.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    arr = np.array(result)
    # Fill color: median of crop (approximates local background).
    fill = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))
    for block in page.text_blocks:
        if not _bboxes_intersect(block.bbox, crop_box_pt):
            continue
        tb = _page_to_pixel_bbox(
            block.bbox,
            page_width_pt=page.width_pt,
            page_height_pt=page.height_pt,
            image_size=image_size,
            pad_px=1,
        )
        # Convert to crop-local coordinates.
        lx0 = max(0, tb[0] - cx0)
        ly0 = max(0, tb[1] - cy0)
        lx1 = min(result.size[0], tb[2] - cx0)
        ly1 = min(result.size[1], tb[3] - cy0)
        if lx1 > lx0 and ly1 > ly0:
            draw.rectangle((lx0, ly0, lx1 - 1, ly1 - 1), fill=fill)
    return result


def apply_raster_fallback(
    page: PageResult,
    element_id: str,
    *,
    reference_page_image: Image.Image,
) -> dict[str, Any] | None:
    """Last resort for **non-text** regions: paste a PDF crop with text glyphs erased.

    Never applied to text elements (would ghost under editable titles).
    """

    if _find_text_block(page, element_id) is not None:
        return None

    bbox: BBox | None = None
    kind = "unknown"
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
    crop = _erase_overlapping_text_from_crop(
        crop,
        crop_bbox_px=bbox_px,
        page=page,
        image_size=reference_page_image.size,
    )
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

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
        "text_glyphs_erased": True,
    }


def propose_fixes_for_issue(issue: RefineIssue) -> list[str]:
    """Ordered fix actions for an issue kind (deterministic priority)."""

    element_kind = str(issue.details.get("element_kind") or "unknown")
    if issue.kind in {"text_overflow", "extra_wrap"}:
        # Never raster-fallback text.
        return ["shrink_font", "widen_bbox", "match_line_bboxes"]
    if issue.kind == "line_drift":
        return ["match_line_bboxes", "widen_bbox", "shrink_font"]
    if issue.kind == "low_local_ssim":
        if element_kind == "text":
            return ["match_line_bboxes", "widen_bbox", "shrink_font"]
        # Non-text visual defects: try raster first (the real defect class).
        return ["raster_fallback"]
    if element_kind == "text":
        return ["shrink_font", "widen_bbox", "match_line_bboxes"]
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
    max_attempts_per_slide: int = DEFAULT_MAX_ATTEMPTS_PER_SLIDE,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    dpi: float = DEFAULT_COMPARE_DPI,
    write_pptx: WritePptxFn | None = None,
    render_pptx: RenderPagesFn | None = None,
    render_pdf_page: RenderPdfPageFn | None = None,
    work_dir: Path | None = None,
) -> tuple[list[PageResult], DeckRefineReport]:
    """Run the render-compare-refine loop.

    Returns the (possibly improved) page list and a deck-level report. When
    LibreOffice is missing or fails, restores the last accepted PPTX and
    returns a skipped report instead of raising.
    """

    from .pipeline import render_pages_to_pptx

    started = perf_counter()
    iters = max(0, int(iterations))
    soffice_calls = 0
    total_attempts = 0

    def _report(
        *,
        working_pages: list[PageResult],
        skipped: str | None,
        page_reports: list[SlideRefineReport] | None = None,
    ) -> tuple[list[PageResult], DeckRefineReport]:
        return working_pages, DeckRefineReport(
            enabled=True,
            skipped_reason=skipped,
            iterations=iters,
            dpi=dpi,
            pages=list(page_reports or []),
            runtime_seconds=round(perf_counter() - started, 3),
            soffice_call_count=soffice_calls,
            attempt_count=total_attempts,
        )

    if iters <= 0:
        return _report(working_pages=pages, skipped="refine_iters<=0")

    if find_libreoffice_executable() is None and render_pptx is None:
        logger.warning(
            "Refine skipped: LibreOffice (soffice) is not available. Install libreoffice-impress to enable --refine."
        )
        return _report(working_pages=pages, skipped="libreoffice_unavailable")

    write_fn = write_pptx or render_pages_to_pptx
    render_pptx_fn = render_pptx or render_pptx_pages
    render_pdf_fn = render_pdf_page or render_pdf_page_image

    working = _deepcopy_pages(pages)
    owns_work = work_dir is None
    root = Path(tempfile.mkdtemp(prefix="pdf2ppt-refine-")) if owns_work else Path(work_dir or ".")
    root.mkdir(parents=True, exist_ok=True)

    def _safe_render(pptx_path: Path, label: str) -> list[Image.Image]:
        nonlocal soffice_calls
        soffice_calls += 1
        return render_pptx_fn(pptx_path, dpi=dpi, work_dir=root / label)

    def _write_full() -> None:
        write_fn(
            working,
            output_pptx,
            slide_width_pt=slide_width_pt,
            slide_height_pt=slide_height_pt,
        )

    def _write_single(page_index: int, path: Path) -> None:
        write_fn(
            [working[page_index]],
            path,
            slide_width_pt=slide_width_pt,
            slide_height_pt=slide_height_pt,
        )

    try:
        # Baseline: write full deck (caller expects a valid PPTX even if refine skips later).
        _write_full()
        try:
            # Prefer per-slide temps for diagnosis/fix trials (cheaper on multi-page).
            baseline_images: list[Image.Image] = []
            for page_index in range(len(working)):
                if perf_counter() - started > time_budget_seconds:
                    break
                single_path = root / f"baseline-p{page_index}.pptx"
                _write_single(page_index, single_path)
                rendered_one = _safe_render(single_path, f"baseline-r{page_index}")
                baseline_images.append(rendered_one[0] if rendered_one else Image.new("RGB", (8, 8)))
        except RenderCompareError as error:
            logger.warning("Refine skipped: LibreOffice render failed (%s)", error)
            _write_full()
            return _report(working_pages=pages, skipped="render_failed")

        page_reports: list[SlideRefineReport] = []

        for page_index, page in enumerate(working):
            if perf_counter() - started > time_budget_seconds:
                page_reports.append(
                    SlideRefineReport(
                        page_number=page.page_number,
                        iterations=0,
                        metric_before=None,
                        metric_after=None,
                        skipped_reason="time_budget",
                    )
                )
                continue
            if page_index >= len(baseline_images):
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

            try:
                reference = render_pdf_fn(pdf_path, page_index, dpi=dpi)
            except Exception as error:  # noqa: BLE001
                logger.warning("Refine page %s: PDF render failed (%s)", page.page_number, error)
                page_reports.append(
                    SlideRefineReport(
                        page_number=page.page_number,
                        iterations=0,
                        metric_before=None,
                        metric_after=None,
                        skipped_reason="render_failed",
                    )
                )
                continue

            single_path = root / f"page-{page_index}.pptx"
            _write_single(page_index, single_path)
            candidate = baseline_images[page_index]
            metric_before, issues = diagnose_slide(reference, candidate, page)
            try:
                edit_before = slide_editability_score(single_path, slide_index=0)
                text_f1_before = slide_text_f1_vs_pdf(single_path, pdf_path, page_index=page_index)
            except Exception:  # noqa: BLE001
                edit_before = 0.0
                text_f1_before = 0.0

            slide_report = SlideRefineReport(
                page_number=page.page_number,
                iterations=0,
                metric_before=metric_before,
                metric_after=metric_before,
                issues=list(issues),
                editability_before=edit_before,
                editability_after=edit_before,
                text_f1_before=text_f1_before,
                text_f1_after=text_f1_before,
            )
            current_metric = metric_before
            current_edit = edit_before
            current_text_f1 = text_f1_before
            fixes_applied = 0
            attempts = 0
            page_soffice_before = soffice_calls
            last_good_digest = pptx_content_digest(single_path)

            # Accepted improving iterations only (no-ops / rejects do not consume).
            improving_iters = 0
            while improving_iters < iters and fixes_applied < max_fixes_per_slide:
                if attempts >= max_attempts_per_slide:
                    break
                if perf_counter() - started > time_budget_seconds:
                    slide_report.skipped_reason = slide_report.skipped_reason or "time_budget"
                    break

                _, issues = diagnose_slide(reference, candidate, page, page_ssim=current_metric)
                if not issues:
                    break

                made_progress = False
                for issue in issues:
                    if attempts >= max_attempts_per_slide or fixes_applied >= max_fixes_per_slide:
                        break
                    for action in propose_fixes_for_issue(issue):
                        if attempts >= max_attempts_per_slide or fixes_applied >= max_fixes_per_slide:
                            break
                        if perf_counter() - started > time_budget_seconds:
                            break

                        snapshot = _deepcopy_pages(working)
                        details = apply_fix_action(
                            working[page_index],
                            element_id=issue.element_id,
                            action=action,
                            reference_page_image=reference,
                        )
                        if details is None:
                            continue

                        attempts += 1
                        total_attempts += 1
                        _write_single(page_index, single_path)
                        trial_digest = pptx_content_digest(single_path)
                        xml_changed = trial_digest != last_good_digest
                        if not xml_changed:
                            # No-op: revert IR, do not consume improving iteration.
                            working[:] = snapshot
                            page = working[page_index]
                            _write_single(page_index, single_path)
                            slide_report.fixes.append(
                                RefineFixAttempt(
                                    issue_kind=issue.kind,
                                    element_id=issue.element_id,
                                    action=action,
                                    accepted=False,
                                    metric_before=current_metric,
                                    metric_after=current_metric,
                                    details={**details, "reject_reason": "xml_unchanged"},
                                    xml_changed=False,
                                    noop=True,
                                )
                            )
                            continue

                        try:
                            trial_rendered = _safe_render(
                                single_path,
                                f"trial-p{page_index}-a{attempts}-{action}",
                            )
                        except RenderCompareError as error:
                            logger.warning(
                                "Refine render failed mid-loop (%s); restoring last accepted state",
                                error,
                            )
                            working[:] = snapshot
                            page = working[page_index]
                            _write_full()
                            slide_report.skipped_reason = "render_failed"
                            slide_report.fixes.append(
                                RefineFixAttempt(
                                    issue_kind=issue.kind,
                                    element_id=issue.element_id,
                                    action=action,
                                    accepted=False,
                                    metric_before=current_metric,
                                    metric_after=None,
                                    details={**details, "reject_reason": "render_failed"},
                                    xml_changed=True,
                                    noop=False,
                                )
                            )
                            # Stop this page; keep prior accepts.
                            made_progress = False
                            improving_iters = iters  # force exit outer while
                            break

                        if not trial_rendered:
                            working[:] = snapshot
                            page = working[page_index]
                            _write_single(page_index, single_path)
                            slide_report.fixes.append(
                                RefineFixAttempt(
                                    issue_kind=issue.kind,
                                    element_id=issue.element_id,
                                    action=action,
                                    accepted=False,
                                    metric_before=current_metric,
                                    metric_after=None,
                                    details={**details, "reject_reason": "missing_rendered_slide"},
                                    xml_changed=True,
                                    noop=False,
                                )
                            )
                            continue

                        trial_candidate = trial_rendered[0]
                        trial_metric = compute_ssim(reference, resize_to_match(reference, trial_candidate))
                        try:
                            trial_edit = slide_editability_score(single_path, slide_index=0)
                            trial_text_f1 = slide_text_f1_vs_pdf(single_path, pdf_path, page_index=page_index)
                        except Exception:  # noqa: BLE001
                            trial_edit = current_edit
                            trial_text_f1 = current_text_f1

                        ssim_improved = trial_metric > current_metric + METRIC_IMPROVE_EPSILON
                        edit_ok = trial_edit + EDITABILITY_EPSILON >= current_edit
                        text_ok = trial_text_f1 + TEXT_F1_EPSILON >= current_text_f1
                        accepted = ssim_improved and edit_ok and text_ok

                        reject_reason = None
                        if not ssim_improved:
                            reject_reason = "ssim_not_improved"
                        elif not edit_ok:
                            reject_reason = "editability_regressed"
                        elif not text_ok:
                            reject_reason = "text_f1_regressed"

                        slide_report.fixes.append(
                            RefineFixAttempt(
                                issue_kind=issue.kind,
                                element_id=issue.element_id,
                                action=action,
                                accepted=accepted,
                                metric_before=current_metric,
                                metric_after=trial_metric,
                                details={
                                    **details,
                                    **({"reject_reason": reject_reason} if reject_reason is not None else {}),
                                    "editability_before": current_edit,
                                    "editability_after": trial_edit,
                                    "text_f1_before": current_text_f1,
                                    "text_f1_after": trial_text_f1,
                                },
                                xml_changed=True,
                                noop=False,
                            )
                        )
                        if accepted:
                            current_metric = trial_metric
                            current_edit = trial_edit
                            current_text_f1 = trial_text_f1
                            candidate = trial_candidate
                            page = working[page_index]
                            last_good_digest = trial_digest
                            fixes_applied += 1
                            improving_iters += 1
                            made_progress = True
                            break

                        working[:] = snapshot
                        page = working[page_index]
                        _write_single(page_index, single_path)
                    else:
                        continue
                    break  # next improving iteration after one accepted issue fix

                if slide_report.skipped_reason == "render_failed":
                    break
                if not made_progress:
                    break

            slide_report.iterations = improving_iters
            slide_report.metric_after = current_metric
            slide_report.editability_after = current_edit
            slide_report.text_f1_after = current_text_f1
            slide_report.attempt_count = attempts
            slide_report.soffice_call_count = soffice_calls - page_soffice_before
            page_reports.append(slide_report)

        # Final write of accepted pages (full deck).
        _write_full()
        return _report(working_pages=working, skipped=None, page_reports=page_reports)
    except RenderCompareError as error:
        logger.warning("Refine skipped: LibreOffice render failed (%s)", error)
        try:
            _write_full()
        except Exception:  # noqa: BLE001
            logger.warning("Could not restore PPTX after refine render failure", exc_info=True)
        return _report(working_pages=pages, skipped="render_failed")
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
    "DEFAULT_MAX_ATTEMPTS_PER_SLIDE",
    "DEFAULT_MAX_FIXES_PER_SLIDE",
    "DEFAULT_REFINE_ITERS",
    "DEFAULT_TIME_BUDGET_SECONDS",
    "METRIC_IMPROVE_EPSILON",
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
    "pptx_content_digest",
    "propose_fixes_for_issue",
    "refine_pages",
    "region_ssim",
    "write_refine_report",
]
