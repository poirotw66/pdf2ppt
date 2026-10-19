"""P2 layout analysis: typed regions + reading order for text grouping.

Two engines share one IR (``LayoutRegion``):

* ``heuristic`` — geometric grouping (default; no model download; used in CI).
* ``ppstructure`` — PaddleOCR ``LayoutDetection`` (PP-DocLayout, the layout stage
  of PP-StructureV3). Soft-falls back to heuristic when the model cannot start.

Downstream, ``apply_layout_to_page`` groups lines into paragraphs/text boxes,
detects bullets, maps titles, crops figures, defers tables to P1, and filters
headers/footers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .models import (
    BBox,
    ImagePlacement,
    LayoutLabel,
    LayoutRegion,
    ShapeElement,
    TableElement,
    TextBlock,
    TextParagraph,
)

logger = logging.getLogger(__name__)

LayoutEngineName = Literal["heuristic", "ppstructure"]

# Process-level LayoutDetection cache so convert_pdf / pages reuse one model load.
_DETECTOR_CACHE: dict[tuple[str, float | None], Any] = {}
_FOOTER_Y0_RATIO = 0.93
_FOOTER_SHAPE_PROXIMITY_PT = 40.0

# PP-DocLayout / PP-StructureV3 label → normalized LayoutLabel.
_PPSTRUCTURE_LABEL_MAP: dict[str, LayoutLabel] = {
    "doc_title": "title",
    "document_title": "title",
    "paragraph_title": "paragraph_title",
    "title": "title",
    "text": "text",
    "content": "text",
    "list": "list",
    "lists": "list",
    "reference": "list",
    "figure": "figure",
    "image": "figure",
    "figure_title": "caption",
    "figure_caption": "caption",
    "fig_title": "caption",
    "caption": "caption",
    "table": "table",
    "table_title": "caption",
    "table_caption": "caption",
    "header": "header",
    "footer": "footer",
    "header_image": "header",
    "footer_image": "footer",
    "number": "page_number",
    "page_number": "page_number",
    "footnote": "footer",
    "formula": "formula",
    "equation": "formula",
    "formula_number": "formula",
    "chart": "chart",
    "seal": "seal",
    "sidebar_text": "sidebar",
    "aside_text": "sidebar",
    "abstract": "text",
    "algorithm": "text",
    "toc": "list",
    "contents": "list",
}

_SKIP_LABELS: frozenset[LayoutLabel] = frozenset({"header", "footer", "page_number", "seal"})
_FIGURE_LABELS: frozenset[LayoutLabel] = frozenset({"figure", "chart"})
_TABLE_LABELS: frozenset[LayoutLabel] = frozenset({"table"})
_TITLE_LABELS: frozenset[LayoutLabel] = frozenset({"title", "paragraph_title"})
_TEXTISH_LABELS: frozenset[LayoutLabel] = frozenset(
    {"title", "paragraph_title", "text", "list", "caption", "formula", "sidebar", "other"}
)

_BULLET_PREFIX_RE = re.compile(
    r"""^(?:
        [\-\*\u2022\u2023\u25E6\u2043\u2219•◦‣∙]
        |\d{1,3}[\.\)]
        |[A-Za-z][\.\)]
        |[ivxlcdmIVXLCDM]{1,6}[\.\)]
    )\s+""",
    re.VERBOSE,
)
_INDENT_LEVEL_CHARS = 18.0  # PDF points of left-indent ≈ one list level on slides


@dataclass(slots=True)
class LayoutPageResult:
    """Output of layout analysis for one page."""

    regions: list[LayoutRegion]
    engine: str
    fallback_reason: str | None = None


@dataclass(slots=True)
class LayoutApplicationResult:
    """Text/image adjustments produced by applying layout regions."""

    text_blocks: list[TextBlock]
    figure_images: list[ImagePlacement] = field(default_factory=list)
    regions: list[LayoutRegion] = field(default_factory=list)
    engine: str = "heuristic"
    fallback_reason: str | None = None
    skipped_block_ids: list[str] = field(default_factory=list)


class LayoutInitializationError(RuntimeError):
    error_code = "layout-initialization-error"

    def to_detail(self) -> dict[str, object]:
        return {"code": self.error_code, "message": str(self)}


def normalize_layout_label(raw_label: str) -> LayoutLabel:
    key = (raw_label or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _PPSTRUCTURE_LABEL_MAP.get(key, "other")


def bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_intersection(a: BBox, b: BBox) -> BBox | None:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def intersection_area(a: BBox, b: BBox) -> float:
    inter = bbox_intersection(a, b)
    return bbox_area(inter) if inter is not None else 0.0


def intersection_over_block(block: BBox, region: BBox) -> float:
    area = bbox_area(block)
    if area <= 0:
        return 0.0
    return intersection_area(block, region) / area


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def union_bbox(boxes: list[BBox]) -> BBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def scale_bbox_from_image_to_page(
    bbox: BBox,
    *,
    image_size: tuple[int, int],
    page_width: float,
    page_height: float,
) -> BBox:
    image_w, image_h = image_size
    scale_x = page_width / max(image_w, 1)
    scale_y = page_height / max(image_h, 1)
    return (
        float(bbox[0]) * scale_x,
        float(bbox[1]) * scale_y,
        float(bbox[2]) * scale_x,
        float(bbox[3]) * scale_y,
    )


def scale_bbox_from_page_to_image(
    bbox: BBox,
    *,
    image_size: tuple[int, int],
    page_width: float,
    page_height: float,
) -> BBox:
    image_w, image_h = image_size
    scale_x = image_w / max(page_width, 1.0)
    scale_y = image_h / max(page_height, 1.0)
    return (
        float(bbox[0]) * scale_x,
        float(bbox[1]) * scale_y,
        float(bbox[2]) * scale_x,
        float(bbox[3]) * scale_y,
    )


def is_ppstructure_available() -> bool:
    """Return True when LayoutDetection can be imported (model still may fail at init)."""

    try:
        from paddleocr import LayoutDetection  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _has_inference_config(directory: Path) -> bool:
    return (directory / "inference.yml").is_file() or (directory / "inference.json").is_file()


def resolve_layout_model_dir(model_root: Path | str | None) -> str | None:
    """Return a concrete ``model_dir`` only when layout weights already exist.

    Passing a bare ``./model`` (OCR root) without DocLayout files makes PaddleX
    look for ``model/inference.yml`` and fail — which then silently falls back
    to heuristic. When nothing is present, return ``None`` so LayoutDetection
    uses its default download cache.
    """

    if model_root is None:
        return None
    root = Path(model_root).expanduser()
    named = root / "PP-DocLayout_plus-L"
    if named.is_dir() and _has_inference_config(named):
        return str(named)
    if root.is_dir() and _has_inference_config(root):
        return str(root)
    return None


def resolve_layout_engine_name(requested: str) -> LayoutEngineName:
    name = (requested or "heuristic").strip().lower()
    if name in {"heuristic", "ppstructure"}:
        return name  # type: ignore[return-value]
    logger.warning("Unknown layout engine %r; using heuristic", requested)
    return "heuristic"


class LayoutEngine:
    """Lazy layout analyzer with heuristic fallback."""

    def __init__(
        self,
        engine: str = "heuristic",
        *,
        model_root: Path | None = Path("model"),
        threshold: float | None = None,
        skip_header_footer: bool = True,
    ) -> None:
        self.requested_engine = resolve_layout_engine_name(engine)
        self.model_root = model_root
        self.threshold = threshold
        self.skip_header_footer = skip_header_footer
        self._detector: Any | None = None
        self._init_error: str | None = None
        self._active_engine: LayoutEngineName = "heuristic"

    @property
    def active_engine(self) -> str:
        return self._active_engine

    def analyze(
        self,
        image: Image.Image | None,
        *,
        page_width: float,
        page_height: float,
        text_blocks: list[TextBlock] | None = None,
        page_number: int = 1,
        shape_bboxes: list[BBox] | None = None,
        table_bboxes: list[BBox] | None = None,
    ) -> LayoutPageResult:
        """Return typed regions in page coordinates (PDF points)."""

        if self.requested_engine == "ppstructure":
            try:
                regions = self._analyze_ppstructure(
                    image,
                    page_width=page_width,
                    page_height=page_height,
                    page_number=page_number,
                )
                self._active_engine = "ppstructure"
                return LayoutPageResult(regions=regions, engine="ppstructure")
            except Exception as error:  # noqa: BLE001
                reason = f"ppstructure unavailable ({error}); fell back to heuristic"
                logger.warning(
                    "Layout engine fallback to heuristic (requested=ppstructure): %s",
                    reason,
                )
                self._init_error = str(error)
                regions = analyze_layout_heuristic(
                    text_blocks or [],
                    page_width=page_width,
                    page_height=page_height,
                    page_number=page_number,
                    shape_bboxes=shape_bboxes,
                    table_bboxes=table_bboxes,
                )
                self._active_engine = "heuristic"
                return LayoutPageResult(regions=regions, engine="heuristic", fallback_reason=reason)

        regions = analyze_layout_heuristic(
            text_blocks or [],
            page_width=page_width,
            page_height=page_height,
            page_number=page_number,
            shape_bboxes=shape_bboxes,
            table_bboxes=table_bboxes,
        )
        self._active_engine = "heuristic"
        return LayoutPageResult(regions=regions, engine="heuristic")

    def _cache_key(self) -> tuple[str, float | None]:
        root = str(Path(self.model_root).expanduser().resolve()) if self.model_root else ""
        return (root, self.threshold)

    def _get_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        cache_key = self._cache_key()
        cached = _DETECTOR_CACHE.get(cache_key)
        if cached is not None:
            self._detector = cached
            return self._detector
        try:
            from paddleocr import LayoutDetection
        except ImportError as error:
            raise LayoutInitializationError(
                "PP-StructureV3 layout requires paddleocr LayoutDetection. "
                "Install paddleocr>=3.4 and a PaddlePaddle runtime, or use "
                "--layout-engine heuristic."
            ) from error

        kwargs: dict[str, Any] = {"layout_nms": True, "model_name": "PP-DocLayout_plus-L"}
        if self.threshold is not None:
            kwargs["threshold"] = self.threshold
        model_dir = resolve_layout_model_dir(self.model_root)
        if model_dir is not None:
            kwargs["model_dir"] = model_dir
        elif self.model_root is not None:
            logger.info(
                "layout_model_root %s has no PP-DocLayout weights; "
                "LayoutDetection will use the PaddleX default download cache",
                self.model_root,
            )
        try:
            self._detector = LayoutDetection(**kwargs)
        except TypeError:
            # Older paddleocr builds may not accept model_dir / model_name together.
            fallback = {key: value for key, value in kwargs.items() if key != "model_dir"}
            try:
                self._detector = LayoutDetection(**fallback)
            except TypeError:
                self._detector = LayoutDetection(layout_nms=True)
        except Exception:
            # Retry without an explicit model name (PaddleX default).
            retry = {key: value for key, value in kwargs.items() if key != "model_name"}
            self._detector = LayoutDetection(**retry)
        _DETECTOR_CACHE[cache_key] = self._detector
        return self._detector

    def _analyze_ppstructure(
        self,
        image: Image.Image | None,
        *,
        page_width: float,
        page_height: float,
        page_number: int,
    ) -> list[LayoutRegion]:
        if image is None:
            raise LayoutInitializationError("ppstructure layout requires a page raster image")
        detector = self._get_detector()
        rgb = image.convert("RGB")
        import numpy as np

        # PaddleX LayoutDetection expects BGR ndarrays by default.
        array = np.ascontiguousarray(np.array(rgb)[:, :, ::-1])
        outputs = list(detector.predict(array, batch_size=1, layout_nms=True))
        if not outputs:
            return []
        payload = _coerce_layout_payload(outputs[0])
        boxes = _extract_layout_boxes(payload)
        regions: list[LayoutRegion] = []
        for index, box in enumerate(boxes):
            raw_label = str(box.get("label") or box.get("cls_label") or "text")
            label = normalize_layout_label(raw_label)
            coordinate = box.get("coordinate") or box.get("bbox") or box.get("box")
            if coordinate is None:
                continue
            coords = [float(value) for value in coordinate[:4]]
            image_bbox: BBox = (coords[0], coords[1], coords[2], coords[3])
            page_bbox = scale_bbox_from_image_to_page(
                image_bbox,
                image_size=rgb.size,
                page_width=page_width,
                page_height=page_height,
            )
            score = float(box.get("score") or box.get("confidence") or 0.0)
            regions.append(
                LayoutRegion(
                    id=f"layout_{page_number}_{index + 1}",
                    label=label,
                    bbox=page_bbox,
                    confidence=score,
                    reading_order=0,
                    raw_label=raw_label,
                    engine="ppstructure",
                )
            )
        return assign_region_reading_order(regions)


def _coerce_layout_payload(result: Any) -> Any:
    if isinstance(result, dict):
        return result
    if hasattr(result, "json"):
        payload = result.json
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict) and "res" in payload:
            return payload["res"]
        return payload
    if hasattr(result, "res"):
        return result.res
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return result


def _extract_layout_boxes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        boxes = payload.get("boxes")
        if isinstance(boxes, list):
            return [box for box in boxes if isinstance(box, dict)]
        for key in ("res", "result", "data", "layout_det_res"):
            nested = payload.get(key)
            if nested is not None:
                found = _extract_layout_boxes(nested)
                if found:
                    return found
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and (
            "label" in payload[0] or "coordinate" in payload[0] or "bbox" in payload[0]
        ):
            return [box for box in payload if isinstance(box, dict)]
        for item in payload:
            found = _extract_layout_boxes(item)
            if found:
                return found
    return []


def assign_region_reading_order(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    """Order regions roughly top-to-bottom, left-to-right (column-aware)."""

    if not regions:
        return regions
    page_width = max((region.bbox[2] for region in regions), default=1.0)
    # Two-column heuristic: split at mid-page when both sides have content.
    mid_x = page_width * 0.5
    left = [region for region in regions if (region.bbox[0] + region.bbox[2]) / 2.0 < mid_x]
    right = [region for region in regions if (region.bbox[0] + region.bbox[2]) / 2.0 >= mid_x]
    use_columns = len(left) >= 2 and len(right) >= 2

    def sort_key(region: LayoutRegion) -> tuple[float, float]:
        return (round(region.bbox[1], 1), round(region.bbox[0], 1))

    ordered: list[LayoutRegion]
    if use_columns:
        ordered = sorted(left, key=sort_key) + sorted(right, key=sort_key)
    else:
        ordered = sorted(regions, key=sort_key)
    for index, region in enumerate(ordered, start=1):
        region.reading_order = index
    return ordered


# ---------------------------------------------------------------------------
# Heuristic layout (CPU / CI fallback)
# ---------------------------------------------------------------------------


def analyze_layout_heuristic(
    text_blocks: list[TextBlock],
    *,
    page_width: float,
    page_height: float,
    page_number: int = 1,
    shape_bboxes: list[BBox] | None = None,
    table_bboxes: list[BBox] | None = None,
) -> list[LayoutRegion]:
    """Build typed regions from existing text blocks without a neural model.

    Groups vertically adjacent lines that share a column into paragraph/list
    regions, promotes large top-of-page text to title, and marks likely
    headers/footers near page edges. Distant same-row labels (flowchart
    ``Collect Info`` vs ``Verify``) stay in separate regions.
    """

    if not text_blocks:
        return []

    sorted_blocks = sorted(text_blocks, key=lambda block: (round(block.bbox[1], 1), round(block.bbox[0], 1)))
    sizes = [block.font_size for block in sorted_blocks if block.font_size]
    baseline = sorted(sizes)[len(sizes) // 2] if sizes else 12.0
    obstacle_bboxes = list(shape_bboxes or []) + list(table_bboxes or [])

    regions: list[LayoutRegion] = []
    consumed: set[str] = set()
    region_index = 0

    for block in sorted_blocks:
        if block.id in consumed:
            continue
        region_index += 1
        label = _heuristic_label_for_block(
            block,
            baseline=baseline,
            page_width=page_width,
            page_height=page_height,
            obstacle_bboxes=obstacle_bboxes,
        )
        members = [block]
        consumed.add(block.id)

        if label in {"text", "list", "caption"}:
            for candidate in sorted_blocks:
                if candidate.id in consumed:
                    continue
                if _can_group_into_paragraph(members[-1], candidate, page_width=page_width):
                    cand_label = _heuristic_label_for_block(
                        candidate,
                        baseline=baseline,
                        page_width=page_width,
                        page_height=page_height,
                        obstacle_bboxes=obstacle_bboxes,
                    )
                    if cand_label in {"text", "list", "caption"} or (
                        label == "list" and cand_label == "list"
                    ):
                        # Keep list vs body consistent within a region.
                        if label == "list" and cand_label != "list" and not _looks_like_list_line(candidate.text):
                            continue
                        if label != "list" and cand_label == "list":
                            label = "list"
                        members.append(candidate)
                        consumed.add(candidate.id)

        region_bbox = union_bbox([member.bbox for member in members])
        if label == "list" or any(_looks_like_list_line(member.text) for member in members):
            label = "list"
        regions.append(
            LayoutRegion(
                id=f"layout_{page_number}_{region_index}",
                label=label,
                bbox=region_bbox,
                confidence=0.55,
                reading_order=0,
                raw_label=f"heuristic:{label}",
                engine="heuristic",
            )
        )

    return assign_region_reading_order(regions)


def _heuristic_label_for_block(
    block: TextBlock,
    *,
    baseline: float,
    page_width: float,
    page_height: float,
    obstacle_bboxes: list[BBox] | None = None,
) -> LayoutLabel:
    text = block.text.strip()
    x0, y0, x1, y1 = block.bbox
    height = max(1.0, y1 - y0)
    width = max(1.0, x1 - x0)
    near_top = y0 < page_height * 0.06
    compact = len(re.sub(r"\s+", "", text)) <= 24
    tiny_font = block.font_size is not None and block.font_size <= baseline * 0.85
    # Absolute small type covers HF when the page median is already tiny (OCR-only pages).
    small_type = tiny_font or (block.font_size is not None and block.font_size <= 11.0)
    page_number_like = bool(text.isdigit() or re.fullmatch(r"\d+\s*/\s*\d+", text))
    # Headers are typically small type in a thin top band — not large slide titles.
    looks_like_running_header = tiny_font or (block.font_size is None and height <= baseline * 1.1)

    if (
        near_top
        and compact
        and width < page_width * 0.45
        and looks_like_running_header
        and block.block_role != "title"
    ):
        return "header"
    # Footer / page-number: deep bottom band + small type (or page-number glyph).
    # Never treat chart axis labels as footers when a shape/table sits just above.
    if y0 >= page_height * _FOOTER_Y0_RATIO and (small_type or page_number_like):
        if not _has_obstacle_within_above(block.bbox, obstacle_bboxes or [], _FOOTER_SHAPE_PROXIMITY_PT):
            if page_number_like:
                return "page_number"
            return "footer"
    if block.block_role == "title" or (
        block.font_size and block.font_size >= baseline * 1.35 and near_top and y0 < page_height * 0.28
    ):
        return "title"
    if _looks_like_list_line(text) or block.block_role == "list":
        return "list"
    if height < baseline * 0.9 and y0 > page_height * 0.5 and width < page_width * 0.55:
        # Likely caption under a figure — weak heuristic only.
        if text.lower().startswith(("fig", "figure", "表", "圖", "图")):
            return "caption"
    return "text"


def _has_obstacle_within_above(
    text_bbox: BBox,
    obstacles: list[BBox],
    proximity_pt: float,
) -> bool:
    """True when a shape/table bottom edge sits within ``proximity_pt`` above the text."""

    tx0, ty0, tx1, _ty1 = text_bbox
    for ox0, _oy0, ox1, oy1 in obstacles:
        horizontal_overlap = min(tx1, ox1) - max(tx0, ox0)
        if horizontal_overlap <= 0:
            continue
        gap = ty0 - oy1
        if 0 <= gap <= proximity_pt:
            return True
    return False


def _looks_like_list_line(text: str) -> bool:
    stripped = text.lstrip()
    return bool(_BULLET_PREFIX_RE.match(stripped))


def _can_group_into_paragraph(
    previous: TextBlock,
    candidate: TextBlock,
    *,
    page_width: float,
) -> bool:
    """True when candidate continues previous as the same paragraph/column.

    Horizontal separation larger than ~18% of page width rejects flowchart
    side-by-side shape labels even when their y-ranges overlap.
    """

    prev = previous.bbox
    cand = candidate.bbox
    prev_height = max(1.0, prev[3] - prev[1])
    cand_height = max(1.0, cand[3] - cand[1])
    avg_height = (prev_height + cand_height) / 2.0

    # Same-row labels with a large horizontal gap stay separate (flowchart bug).
    vertical_overlap = max(0.0, min(prev[3], cand[3]) - max(prev[1], cand[1]))
    min_height = min(prev_height, cand_height)
    if min_height > 0 and vertical_overlap / min_height >= 0.45:
        gap = max(0.0, max(cand[0] - prev[2], prev[0] - cand[2]))
        if gap > page_width * 0.08:
            return False
        # Same line continuation (wrapped? unlikely) — only merge tiny gaps.
        return gap <= avg_height * 0.6

    # Stacked in the same column.
    vertical_gap = cand[1] - prev[3]
    if vertical_gap < -avg_height * 0.2 or vertical_gap > avg_height * 1.35:
        return False
    prev_center_x = (prev[0] + prev[2]) / 2.0
    cand_center_x = (cand[0] + cand[2]) / 2.0
    if abs(prev_center_x - cand_center_x) > page_width * 0.18:
        return False
    left_delta = abs(prev[0] - cand[0])
    if left_delta > page_width * 0.12 and not _looks_like_list_line(candidate.text):
        return False
    return True


# ---------------------------------------------------------------------------
# Apply regions → regrouped text boxes / figures / skip headers
# ---------------------------------------------------------------------------


def apply_layout_to_page(
    text_blocks: list[TextBlock],
    layout: LayoutPageResult,
    *,
    page_width: float,
    page_height: float,
    page_image: Image.Image | None = None,
    table_elements: list[TableElement] | None = None,
    shape_elements: list[ShapeElement] | None = None,
    skip_header_footer: bool = True,
    page_number: int = 1,
) -> LayoutApplicationResult:
    """Group text by layout regions and emit figure crops when possible."""

    text_blocks = split_wide_multiline_blocks(text_blocks, page_width=page_width)
    regions = list(layout.regions)
    tables = list(table_elements or [])
    shapes = list(shape_elements or [])
    if not regions:
        # Still run list/title heuristics on individual blocks.
        enriched = [_enrich_block_paragraphs(block) for block in text_blocks]
        _assign_title_placeholder(enriched, page_height=page_height)
        return LayoutApplicationResult(
            text_blocks=_reindex_reading_order(enriched),
            regions=[],
            engine=layout.engine,
            fallback_reason=layout.fallback_reason,
        )

    assigned: dict[str, list[TextBlock]] = {region.id: [] for region in regions}
    unassigned: list[TextBlock] = []
    skipped: list[str] = []

    for block in text_blocks:
        if _block_inside_native_table(block, tables):
            skipped.append(block.id)
            continue
        best_region, score = _best_region_for_block(block, regions)
        if best_region is None or score < 0.35:
            unassigned.append(block)
            continue
        if skip_header_footer and best_region.label in _SKIP_LABELS:
            # OCR HF stays in the raster background. Native digital text has no
            # raster fallback, so keep it visible as a text box.
            if block.source == "ocr":
                skipped.append(block.id)
                continue
            block.block_role = best_region.label
            block.layout_label = best_region.label
            unassigned.append(block)
            continue
        if best_region.label in _TABLE_LABELS and tables:
            # Defer to P1 native tables — do not also emit a text box.
            if any(intersection_over_block(block.bbox, table.bbox) >= 0.5 for table in tables):
                skipped.append(block.id)
                continue
        if best_region.label in _FIGURE_LABELS:
            # Text sitting on a figure is usually a label; keep it unless it is
            # almost entirely covered and looks like OCR noise inside the art.
            if intersection_over_block(block.bbox, best_region.bbox) >= 0.9 and len(block.text.strip()) <= 2:
                skipped.append(block.id)
                continue
            unassigned.append(block)
            continue
        assigned[best_region.id].append(block)

    grouped_blocks: list[TextBlock] = []
    for region in sorted(regions, key=lambda item: item.reading_order):
        members = assigned.get(region.id) or []
        if not members:
            continue
        if region.label in _SKIP_LABELS and skip_header_footer:
            continue
        if region.label in _TABLE_LABELS:
            if tables:
                continue
            # Scanned / undetected tables: keep one box per line (no mega-merge).
            for member in members:
                grouped_blocks.append(_enrich_block_paragraphs(member))
            continue
        grouped_blocks.extend(
            _group_region_members(
                members,
                region=region,
                page_number=page_number,
            )
        )

    for block in unassigned:
        grouped_blocks.append(_enrich_block_paragraphs(block))

    _assign_title_placeholder(grouped_blocks, page_height=page_height)
    grouped_blocks = _reindex_reading_order(grouped_blocks)

    figure_images = extract_figure_image_placements(
        regions,
        page_image=page_image,
        page_width=page_width,
        page_height=page_height,
        shape_elements=shapes,
        table_elements=tables,
    )

    return LayoutApplicationResult(
        text_blocks=grouped_blocks,
        figure_images=figure_images,
        regions=regions,
        engine=layout.engine,
        fallback_reason=layout.fallback_reason,
        skipped_block_ids=skipped,
    )


def extract_figure_image_placements(
    regions: list[LayoutRegion],
    *,
    page_image: Image.Image | None,
    page_width: float,
    page_height: float,
    shape_elements: list[ShapeElement] | None = None,
    table_elements: list[TableElement] | None = None,
) -> list[ImagePlacement]:
    if page_image is None:
        return []
    placements: list[ImagePlacement] = []
    rgb = page_image.convert("RGB")
    native_obstacles = [shape.bbox for shape in (shape_elements or [])] + [
        table.bbox for table in (table_elements or [])
    ]
    for region in regions:
        if region.label not in _FIGURE_LABELS:
            continue
        if _figure_overlaps_native(region.bbox, native_obstacles):
            # Region already nativized as vectors/tables — cropping again doubles ink.
            continue
        image_bbox = scale_bbox_from_page_to_image(
            region.bbox,
            image_size=rgb.size,
            page_width=page_width,
            page_height=page_height,
        )
        crop = _safe_crop(rgb, image_bbox)
        if crop is None:
            continue
        from .native_extraction import pil_to_png_bytes

        placements.append(ImagePlacement(bbox=region.bbox, png_bytes=pil_to_png_bytes(crop)))
    return placements


def _figure_overlaps_native(figure_bbox: BBox, native_bboxes: list[BBox]) -> bool:
    """Skip figure crops that largely cover already-nativized shapes/tables."""

    figure_area = bbox_area(figure_bbox)
    if figure_area <= 0 or not native_bboxes:
        return False
    covered = 0.0
    for native in native_bboxes:
        covered += intersection_area(figure_bbox, native)
    # Any substantial overlap with native content means the figure is redundant.
    return covered / figure_area >= 0.25 or any(
        intersection_area(figure_bbox, native) / max(bbox_area(native), 1.0) >= 0.5 for native in native_bboxes
    )


def _safe_crop(image: Image.Image, bbox: BBox) -> Image.Image | None:
    x0 = max(0, int(bbox[0]))
    y0 = max(0, int(bbox[1]))
    x1 = min(image.width, int(round(bbox[2])))
    y1 = min(image.height, int(round(bbox[3])))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return image.crop((x0, y0, x1, y1))


def _best_region_for_block(
    block: TextBlock,
    regions: list[LayoutRegion],
) -> tuple[LayoutRegion | None, float]:
    best: LayoutRegion | None = None
    best_score = 0.0
    for region in regions:
        score = intersection_over_block(block.bbox, region.bbox)
        # Prefer tighter textish regions over huge figure/table containers
        # when scores are close.
        if region.label in _FIGURE_LABELS or region.label in _TABLE_LABELS:
            score *= 0.85
        if score > best_score:
            best_score = score
            best = region
    return best, best_score


def _block_inside_native_table(block: TextBlock, tables: list[TableElement]) -> bool:
    for table in tables:
        if block.id in {cell_id for cell in table.cells for cell_id in cell.text_block_ids}:
            return True
        if intersection_over_block(block.bbox, table.bbox) >= 0.6:
            return True
    return False


def _group_region_members(
    members: list[TextBlock],
    *,
    region: LayoutRegion,
    page_number: int,
) -> list[TextBlock]:
    ordered = sorted(members, key=lambda block: (round(block.bbox[1], 1), round(block.bbox[0], 1)))
    role = _role_for_label(region.label)
    paragraphs = _paragraphs_from_members(ordered, region_label=region.label)
    bbox = union_bbox([block.bbox for block in ordered])
    text = "\n".join(paragraph.text for paragraph in paragraphs if paragraph.text.strip())
    base = ordered[0]
    font_sizes = [
        block.font_size if block.font_size is not None else _estimate_font_size_from_bbox(block.bbox)
        for block in ordered
    ]
    font_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else base.font_size
    confidence = sum(block.confidence for block in ordered) / max(len(ordered), 1)
    use_title = region.label in _TITLE_LABELS
    # Union OCR erase masks so every grouped line is inpainted (avoids ghost text).
    image_boxes = [block.image_bbox for block in ordered if block.image_bbox is not None]
    merged_image_bbox = union_bbox(image_boxes) if image_boxes else base.image_bbox
    return [
        TextBlock(
            id=f"layout_text_{page_number}_{region.reading_order}",
            source=base.source,
            bbox=bbox,
            text=text,
            confidence=confidence,
            font_family=base.font_family,
            font_size=font_size,
            font_color=base.font_color,
            # Preserve PDF boldness — never force titles bold.
            bold=any(block.bold for block in ordered),
            italic=any(block.italic for block in ordered),
            reading_order=region.reading_order,
            block_role=role,
            image_bbox=merged_image_bbox,
            image_polygon=None if image_boxes else base.image_polygon,
            layout_region_id=region.id,
            layout_label=region.label,
            paragraphs=paragraphs,
            use_title_placeholder=use_title,
        )
    ]


def _estimate_font_size_from_bbox(bbox: BBox) -> float:
    return max(6.0, (bbox[3] - bbox[1]) * 0.75)


def _role_for_label(label: LayoutLabel) -> str:
    if label in _TITLE_LABELS:
        return "title"
    if label == "list":
        return "list"
    if label == "caption":
        return "caption"
    if label in _SKIP_LABELS:
        return label
    return "body"


def _paragraphs_from_members(
    members: list[TextBlock],
    *,
    region_label: LayoutLabel,
) -> list[TextParagraph]:
    paragraphs: list[TextParagraph] = []
    left_edges = [block.bbox[0] for block in members]
    min_left = min(left_edges) if left_edges else 0.0
    previous_bottom: float | None = None
    for block in members:
        for line in block.text.splitlines() or [block.text]:
            raw = line.strip()
            if not raw:
                continue
            is_list = region_label == "list" or _looks_like_list_line(raw)
            cleaned, detected_level, list_style = strip_bullet_prefix(
                raw, left=block.bbox[0], min_left=min_left
            )
            list_level: int | None = detected_level if is_list else None
            font_size = block.font_size
            if font_size is None:
                font_size = _estimate_font_size_from_bbox(block.bbox)
            space_before: float | None = None
            if previous_bottom is not None:
                gap = block.bbox[1] - previous_bottom
                if gap > 1.0:
                    space_before = gap
            paragraphs.append(
                TextParagraph(
                    text=cleaned if is_list else raw,
                    list_level=list_level,
                    list_style=list_style if is_list else None,
                    bold=block.bold,
                    italic=block.italic,
                    font_size=font_size,
                    font_color=block.font_color,
                    space_before_pt=space_before,
                )
            )
            previous_bottom = block.bbox[3]
    if not paragraphs:
        paragraphs.append(TextParagraph(text=""))
    return paragraphs


def strip_bullet_prefix(text: str, *, left: float, min_left: float) -> tuple[str, int, str]:
    """Remove literal bullet/number markers and estimate indent level + list style."""

    match = _BULLET_PREFIX_RE.match(text)
    cleaned = text[match.end() :].strip() if match else text.strip()
    indent = max(0.0, left - min_left)
    # Floor (not round) so a modest nested indent maps to level 1, not 2.
    level = int(indent // _INDENT_LEVEL_CHARS)
    # Numbered "1." / "a." at the same indent stays level 0; nested indents rise.
    level = max(0, min(level, 8))
    list_style = "bullet"
    if match:
        token = match.group(0).strip()
        if re.match(r"^\d", token):
            list_style = "arabicPeriod"
        elif re.match(r"^[A-Z][\.\)]", token):
            list_style = "alphaUcPeriod"
        elif re.match(r"^[a-z][\.\)]", token):
            list_style = "alphaLcPeriod"
    return cleaned or text.strip(), level, list_style


def _enrich_block_paragraphs(block: TextBlock) -> TextBlock:
    """Ensure paragraphs/list metadata exist for ungrouped blocks."""

    if block.paragraphs:
        return block
    lines = [line for line in block.text.splitlines() if line.strip()] or [block.text]
    if len(lines) == 1 and not _looks_like_list_line(lines[0]):
        return block
    min_left = block.bbox[0]
    paragraphs: list[TextParagraph] = []
    any_list = False
    for line in lines:
        raw = line.strip()
        is_list = _looks_like_list_line(raw)
        any_list = any_list or is_list
        cleaned, level, list_style = strip_bullet_prefix(raw, left=block.bbox[0], min_left=min_left)
        paragraphs.append(
            TextParagraph(
                text=cleaned if is_list else raw,
                list_level=level if is_list else None,
                list_style=list_style if is_list else None,
                bold=block.bold,
                italic=block.italic,
                font_size=block.font_size or _estimate_font_size_from_bbox(block.bbox),
                font_color=block.font_color,
            )
        )
    block.paragraphs = paragraphs
    if any_list:
        block.block_role = "list"
        block.text = "\n".join(paragraph.text for paragraph in paragraphs)
    return block


def _assign_title_placeholder(blocks: list[TextBlock], *, page_height: float) -> None:
    title_candidates = [
        block
        for block in blocks
        if block.block_role == "title" or block.layout_label in _TITLE_LABELS or block.use_title_placeholder
    ]
    if not title_candidates:
        return
    # Prefer the top-most large title; only one placeholder per slide.
    title_candidates.sort(key=lambda block: (block.bbox[1], -(block.font_size or 0.0)))
    chosen = title_candidates[0]
    if chosen.bbox[1] > page_height * 0.4:
        return
    for block in blocks:
        block.use_title_placeholder = block.id == chosen.id


def _reindex_reading_order(blocks: list[TextBlock]) -> list[TextBlock]:
    ordered = sorted(
        blocks,
        key=lambda block: (
            block.reading_order if block.reading_order else 10_000,
            round(block.bbox[1], 1),
            round(block.bbox[0], 1),
        ),
    )
    for index, block in enumerate(ordered, start=1):
        block.reading_order = index
    return ordered


def split_wide_multiline_blocks(
    blocks: list[TextBlock],
    *,
    page_width: float,
) -> list[TextBlock]:
    """Defense-in-depth: split leftover wide multi-line boxes into per-line boxes.

    Native extraction usually handles this; this covers OCR or older IR that still
    carries ``Collect Info\\nVerify`` style labels in one bbox.
    """

    result: list[TextBlock] = []
    for block in blocks:
        lines = [line for line in block.text.splitlines() if line.strip()]
        width = max(0.0, block.bbox[2] - block.bbox[0])
        if len(lines) < 2 or width < page_width * 0.25:
            result.append(block)
            continue
        # Estimate equal-width slices only when lines are short labels.
        if any(len(line) > 40 for line in lines):
            result.append(block)
            continue
        slice_width = width / len(lines)
        x0 = block.bbox[0]
        for index, line in enumerate(lines):
            line_bbox = (
                x0 + index * slice_width,
                block.bbox[1],
                x0 + (index + 1) * slice_width,
                block.bbox[3],
            )
            result.append(
                TextBlock(
                    id=f"{block.id}_split_{index}",
                    source=block.source,
                    bbox=line_bbox,
                    text=line.strip(),
                    confidence=block.confidence,
                    font_family=block.font_family,
                    font_size=block.font_size,
                    font_color=block.font_color,
                    bold=block.bold,
                    italic=block.italic,
                    reading_order=block.reading_order,
                    block_role=block.block_role,
                    image_bbox=block.image_bbox,
                    image_polygon=block.image_polygon,
                )
            )
    return result


def parse_layout_regions_from_dict(payload: list[dict[str, Any]], *, page_number: int = 1) -> list[LayoutRegion]:
    """Build regions from a mocked / serialized PP-Structure-like box list."""

    regions: list[LayoutRegion] = []
    for index, box in enumerate(payload):
        raw_label = str(box.get("label") or "text")
        coordinate = box.get("coordinate") or box.get("bbox") or [0, 0, 0, 0]
        coords = [float(value) for value in list(coordinate)[:4]]
        regions.append(
            LayoutRegion(
                id=str(box.get("id") or f"layout_{page_number}_{index + 1}"),
                label=normalize_layout_label(raw_label),
                bbox=(coords[0], coords[1], coords[2], coords[3]),
                confidence=float(box.get("score") or box.get("confidence") or 1.0),
                reading_order=int(box.get("reading_order") or 0),
                raw_label=raw_label,
                engine=str(box.get("engine") or "ppstructure"),
            )
        )
    return assign_region_reading_order(regions)
