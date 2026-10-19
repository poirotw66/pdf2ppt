"""P2 layout-quality metrics: grouping, list detection, title detection.

These metrics compare conversion outputs (``PageResult`` text boxes / PPTX
shapes) against optional ground-truth JSON fixtures. They are independent of
LibreOffice fidelity scoring so CI can exercise them without soffice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .layout import bbox_area, intersection_area, iou
from .models import BBox, PageResult, TextBlock


@dataclass(slots=True)
class LayoutGroupingMetrics:
    """Paragraph/text-box grouping quality vs ground truth."""

    predicted_box_count: int
    ground_truth_paragraph_count: int
    box_count_ratio: float
    mean_paragraph_iou: float
    merge_error_count: int
    split_error_count: int
    unmatched_prediction_count: int
    unmatched_ground_truth_count: int
    list_precision: float | None = None
    list_recall: float | None = None
    list_f1: float | None = None
    title_detected: bool | None = None
    title_expected: bool | None = None
    title_correct: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GroundTruthParagraph:
    id: str
    bbox: BBox
    text: str = ""
    role: str = "body"  # title | body | list | caption | header | footer
    list_level: int | None = None


@dataclass(slots=True)
class LayoutGroundTruth:
    page_number: int = 1
    paragraphs: list[GroundTruthParagraph] = field(default_factory=list)
    expect_title: bool = False
    expect_list_item_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "expect_title": self.expect_title,
            "expect_list_item_count": self.expect_list_item_count,
            "paragraphs": [
                {
                    "id": item.id,
                    "bbox": list(item.bbox),
                    "text": item.text,
                    "role": item.role,
                    "list_level": item.list_level,
                }
                for item in self.paragraphs
            ],
        }


def load_layout_ground_truth(path: Path) -> LayoutGroundTruth:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paragraphs = [
        GroundTruthParagraph(
            id=str(item.get("id") or f"gt_{index}"),
            bbox=_as_bbox(item["bbox"]),
            text=str(item.get("text") or ""),
            role=str(item.get("role") or "body"),
            list_level=item.get("list_level"),
        )
        for index, item in enumerate(payload.get("paragraphs") or [])
    ]
    return LayoutGroundTruth(
        page_number=int(payload.get("page_number") or 1),
        paragraphs=paragraphs,
        expect_title=bool(payload.get("expect_title")),
        expect_list_item_count=payload.get("expect_list_item_count"),
    )


def _as_bbox(raw: Any) -> BBox:
    values = [float(value) for value in list(raw)[:4]]
    return (values[0], values[1], values[2], values[3])


def evaluate_text_grouping(
    predicted_blocks: list[TextBlock],
    ground_truth: LayoutGroundTruth,
    *,
    match_iou_threshold: float = 0.35,
) -> LayoutGroupingMetrics:
    """Score text-box grouping against ground-truth paragraphs."""

    preds = list(predicted_blocks)
    gts = list(ground_truth.paragraphs)
    pred_count = len(preds)
    gt_count = len(gts)
    ratio = (pred_count / gt_count) if gt_count else (0.0 if pred_count == 0 else float("inf"))

    # Greedy IoU matching.
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            score = iou(pred.bbox, gt.bbox)
            if score >= match_iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    for score, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_ious.append(score)

    merge_errors = 0
    split_errors = 0
    for pi, pred in enumerate(preds):
        covering = [
            gi
            for gi, gt in enumerate(gts)
            if intersection_area(pred.bbox, gt.bbox) / max(bbox_area(gt.bbox), 1.0) >= 0.5
        ]
        if len(covering) >= 2:
            merge_errors += 1
    for gi, gt in enumerate(gts):
        covering = [
            pi
            for pi, pred in enumerate(preds)
            if intersection_area(pred.bbox, gt.bbox) / max(bbox_area(pred.bbox), 1.0) >= 0.5
        ]
        if len(covering) >= 2:
            split_errors += 1

    list_precision, list_recall, list_f1 = _list_detection_scores(preds, gts)
    title_detected = any(
        block.block_role == "title" or block.use_title_placeholder or block.layout_label in {"title", "paragraph_title"}
        for block in preds
    )
    title_expected = ground_truth.expect_title or any(item.role == "title" for item in gts)
    title_correct: bool | None
    if not title_expected and not title_detected:
        title_correct = True
    elif title_expected:
        title_correct = title_detected
    else:
        title_correct = False

    return LayoutGroupingMetrics(
        predicted_box_count=pred_count,
        ground_truth_paragraph_count=gt_count,
        box_count_ratio=round(ratio, 6) if ratio != float("inf") else -1.0,
        mean_paragraph_iou=round(sum(matched_ious) / len(matched_ious), 6) if matched_ious else 0.0,
        merge_error_count=merge_errors,
        split_error_count=split_errors,
        unmatched_prediction_count=pred_count - len(matched_pred),
        unmatched_ground_truth_count=gt_count - len(matched_gt),
        list_precision=list_precision,
        list_recall=list_recall,
        list_f1=list_f1,
        title_detected=title_detected,
        title_expected=title_expected,
        title_correct=title_correct,
    )


def evaluate_page_layout(
    page_result: PageResult,
    ground_truth: LayoutGroundTruth,
) -> LayoutGroupingMetrics:
    return evaluate_text_grouping(page_result.text_blocks, ground_truth)


def _list_detection_scores(
    preds: list[TextBlock],
    gts: list[GroundTruthParagraph],
) -> tuple[float | None, float | None, float | None]:
    gt_list = [item for item in gts if item.role == "list"]
    pred_items = _expand_predicted_list_items(preds)
    if not gt_list and not pred_items:
        return None, None, None

    gt_items = len(gt_list)
    if gt_items == 0 and not pred_items:
        return None, None, None

    # Match predicted list *items* (often nested in one text box) to GT items.
    matched = 0
    used_gt: set[int] = set()
    for pred_bbox, _pred_text in pred_items:
        best_gi = None
        best_score = 0.0
        for gi, gt in enumerate(gt_list):
            if gi in used_gt:
                continue
            score = iou(pred_bbox, gt.bbox)
            # Also accept containment when a single list box covers many items.
            containment = intersection_area(pred_bbox, gt.bbox) / max(bbox_area(gt.bbox), 1.0)
            score = max(score, containment * 0.9)
            if score > best_score:
                best_score = score
                best_gi = gi
        if best_gi is not None and best_score >= 0.25:
            used_gt.add(best_gi)
            matched += 1

    pred_count = max(len(pred_items), 1) if pred_items else 0
    precision = matched / pred_count if pred_items else 0.0
    recall = matched / max(gt_items, 1) if gt_items else (1.0 if not pred_items else 0.0)
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 6), round(recall, 6), round(f1, 6)


def _expand_predicted_list_items(preds: list[TextBlock]) -> list[tuple[BBox, str]]:
    items: list[tuple[BBox, str]] = []
    for block in preds:
        paragraphs = block.paragraphs or []
        list_paras = [para for para in paragraphs if para.list_level is not None]
        if list_paras:
            # Distribute the parent bbox into horizontal bands for IoU matching.
            band_height = max(1.0, (block.bbox[3] - block.bbox[1]) / max(len(list_paras), 1))
            for index, para in enumerate(list_paras):
                y0 = block.bbox[1] + index * band_height
                y1 = y0 + band_height
                items.append(((block.bbox[0], y0, block.bbox[2], y1), para.text))
            continue
        if block.block_role == "list":
            lines = [line for line in block.text.splitlines() if line.strip()] or [block.text]
            band_height = max(1.0, (block.bbox[3] - block.bbox[1]) / max(len(lines), 1))
            for index, line in enumerate(lines):
                y0 = block.bbox[1] + index * band_height
                y1 = y0 + band_height
                items.append(((block.bbox[0], y0, block.bbox[2], y1), line))
    return items


def count_pptx_bullet_paragraphs(pptx_path: Path) -> int:
    """Count paragraphs that have a real bullet (buChar/buAutoNum), not literal glyphs."""

    from pptx import Presentation
    from pptx.oxml.ns import qn

    presentation = Presentation(str(pptx_path))
    count = 0
    bullet_tags = {qn("a:buChar"), qn("a:buAutoNum"), qn("a:buBlip")}
    for slide in presentation.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:
                try:
                    p_pr = paragraph._p.pPr  # noqa: SLF001
                except Exception:  # noqa: BLE001
                    continue
                if p_pr is None:
                    continue
                if any(child.tag in bullet_tags for child in p_pr):
                    count += 1
    return count


def slide_has_title_placeholder_text(pptx_path: Path) -> bool:
    from pptx import Presentation

    presentation = Presentation(str(pptx_path))
    for slide in presentation.slides:
        title = slide.shapes.title
        if title is not None and (title.text or "").strip():
            return True
    return False
