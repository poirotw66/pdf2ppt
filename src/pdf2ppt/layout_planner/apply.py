"""Apply a validated layout plan onto geometric text blocks / SlideIR groups.

Default application is **low-risk only**: roles, reading order, title placeholder,
and header/footer filtering layered on top of **heuristic** paragraph/list
grouping (see ``annotate_grouped_blocks_with_plan``). Grouping merges and
font-class remaps are opt-in (``apply_grouping`` / ``apply_fonts``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..layout import (
    _assign_title_placeholder,
    _enrich_block_paragraphs,
    _estimate_font_size_from_bbox,
    _paragraphs_from_members,
    intersection_area,
    strip_bullet_prefix,
    union_bbox,
)
from ..models import BBox, GroupElement, LayoutRegion, TextBlock, TextParagraph
from .fonts import is_known_font_family, normalize_font_map, resolve_font_family
from .types import (
    SKIPPABLE_ROLES,
    TITLE_ROLES,
    LayoutPlan,
    LayoutPlannerApplication,
    PlannedBlock,
    PlannedGroup,
)

# Map planner roles onto layout labels / block_role strings used by ppt_render.
_ROLE_TO_LAYOUT_LABEL: dict[str, str] = {
    "title": "title",
    "subtitle": "paragraph_title",
    "body": "text",
    "list_item": "list",
    "caption": "caption",
    "header": "header",
    "footer": "footer",
    "page_number": "page_number",
    "table_cell": "table",
    "other": "other",
}

_ROLE_TO_BLOCK_ROLE: dict[str, str] = {
    "title": "title",
    "subtitle": "body",
    "body": "body",
    "list_item": "list",
    "caption": "caption",
    "header": "header",
    "footer": "footer",
    "page_number": "page_number",
    "table_cell": "body",
    "other": "body",
}

# Geometric merge guards (relative to median line height / page width).
_MAX_VERTICAL_GAP_RATIO = 1.35
_MAX_LEFT_DELTA_RATIO = 0.08
_MAX_FONT_SIZE_RATIO = 1.35
_MIN_COLUMN_OVERLAP_RATIO = 0.45
_UNSAFE_GROUP_ROLES: frozenset[str] = frozenset({"table_cell", "other"})

# Prefer these roles when aggregating annotations onto a heuristic frame.
_ROLE_PRIORITY: dict[str, int] = {
    "title": 100,
    "subtitle": 80,
    "list_item": 70,
    "caption": 60,
    "header": 50,
    "footer": 50,
    "page_number": 50,
    "body": 20,
    "table_cell": 10,
    "other": 0,
}


def apply_layout_plan(
    text_blocks: list[TextBlock],
    plan: LayoutPlan,
    *,
    page_number: int = 1,
    page_height: float = 540.0,
    page_width: float = 960.0,
    skip_header_footer: bool = True,
    font_map: Mapping[str, Mapping[str, str]] | Mapping[str, str] | None = None,
    apply_grouping: bool = False,
    apply_fonts: bool = False,
    table_bboxes: Sequence[BBox] | None = None,
    chart_bboxes: Sequence[BBox] | None = None,
) -> LayoutPlannerApplication:
    """Annotate (and optionally merge) blocks per plan. Never invents text.

    When ``apply_grouping`` is False, returns ``frames_applied=False`` so the
    caller can keep heuristic paragraph/list grouping and then call
    ``annotate_grouped_blocks_with_plan``.
    """

    resolved_font_map = normalize_font_map(font_map)
    if not apply_grouping:
        plan_metadata: dict[str, Any] = {
            "engine": "layout_planner",
            "model": plan.model,
            "prompt_version": plan.prompt_version,
            "cache_hit": plan.cache_hit,
            "latency_ms": plan.latency_ms,
            "estimated_cost_usd": plan.estimated_cost_usd,
            "block_count": len(plan.blocks),
            "group_count": len(plan.groups),
            "accepted_group_count": 0,
            "apply_grouping": False,
            "apply_fonts": apply_fonts,
            "frames_applied": False,
            "rejected_groups": [],
            "notes": list(plan.notes) + ["defer_grouping_to_heuristic"],
            "plan": plan.to_dict(),
            "layout_regions": [],
        }
        return LayoutPlannerApplication(
            text_blocks=list(text_blocks),
            group_elements=[],
            plan_metadata=plan_metadata,
            used_planner=True,
            fallback_reason=None,
            plan=plan,
            frames_applied=False,
        )

    by_id = {block.id: block for block in text_blocks}
    annotations = {item.id: item for item in plan.blocks}
    obstacle_bboxes = list(table_bboxes or []) + list(chart_bboxes or [])

    accepted_groups: list[PlannedGroup] = []
    rejected_group_notes: list[str] = []
    claimed: set[str] = set()

    for group in sorted(plan.groups, key=lambda item: item.reading_order):
        members = [by_id[member_id] for member_id in group.member_ids if member_id in by_id]
        if len(members) <= 1:
            # Singletons stay on the normal per-block path (no pin_font_size).
            continue
        ok, reason = _group_is_safe_to_merge(
            group,
            members,
            annotations=annotations,
            page_width=page_width,
            all_blocks=text_blocks,
            obstacle_bboxes=obstacle_bboxes,
        )
        if not ok:
            rejected_group_notes.append(f"{group.id}: {reason}")
            continue
        accepted_groups.append(group)
        claimed.update(group.member_ids)

    output_blocks: list[TextBlock] = []
    group_elements: list[GroupElement] = []
    regions: list[LayoutRegion] = []

    for group in accepted_groups:
        members = [by_id[member_id] for member_id in group.member_ids if member_id in by_id]
        if not members:
            continue
        if skip_header_footer and group.role in SKIPPABLE_ROLES:
            if all(member.source == "ocr" for member in members):
                continue
        merged = _merge_members(
            members,
            annotations=annotations,
            role=group.role,
            reading_order=group.reading_order,
            group_id=group.id,
            page_number=page_number,
            font_map=resolved_font_map,
            apply_fonts=apply_fonts,
        )
        output_blocks.append(merged)
        group_elements.append(
            GroupElement(
                id=group.id,
                member_ids=list(group.member_ids),
                bbox=merged.bbox,
                provenance="layout_planner",
                confidence=group.confidence,
                locked=False,
            )
        )
        regions.append(
            LayoutRegion(
                id=f"plan_region_{group.id}",
                label=_ROLE_TO_LAYOUT_LABEL.get(group.role, "other"),  # type: ignore[arg-type]
                bbox=merged.bbox,
                confidence=group.confidence,
                reading_order=group.reading_order,
                raw_label=group.role,
                engine="layout_planner",
            )
        )

    for block in text_blocks:
        if block.id in claimed:
            continue
        annotation = annotations.get(block.id)
        role = annotation.role if annotation is not None else "body"
        if skip_header_footer and role in SKIPPABLE_ROLES and block.source == "ocr":
            continue
        output_blocks.append(
            _annotate_singleton(
                block,
                annotation=annotation,
                font_map=resolved_font_map,
                apply_fonts=apply_fonts,
            )
        )

    output_blocks.sort(key=lambda item: (item.reading_order, item.bbox[1], item.bbox[0]))
    for index, block in enumerate(output_blocks):
        block.reading_order = index
    _assign_title_placeholder(output_blocks, page_height=page_height)
    for block in output_blocks:
        lines = [line for line in (block.text or "").splitlines() if line.strip()]
        if block.use_title_placeholder and len(lines) > 1:
            block.use_title_placeholder = False

    plan_metadata = {
        "engine": "layout_planner",
        "model": plan.model,
        "prompt_version": plan.prompt_version,
        "cache_hit": plan.cache_hit,
        "latency_ms": plan.latency_ms,
        "estimated_cost_usd": plan.estimated_cost_usd,
        "block_count": len(plan.blocks),
        "group_count": len(plan.groups),
        "accepted_group_count": len(accepted_groups),
        "apply_grouping": True,
        "apply_fonts": apply_fonts,
        "frames_applied": True,
        "rejected_groups": rejected_group_notes,
        "notes": list(plan.notes),
        "plan": plan.to_dict(),
        "layout_regions": [region.to_dict() for region in regions],
    }
    return LayoutPlannerApplication(
        text_blocks=output_blocks,
        group_elements=group_elements,
        plan_metadata=plan_metadata,
        used_planner=True,
        fallback_reason=None,
        plan=plan,
        frames_applied=True,
    )


def _normalize_text_chars(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def planner_text_regression(
    before_blocks: Sequence[TextBlock],
    after_blocks: Sequence[TextBlock],
    plan: LayoutPlan,
) -> str | None:
    """Return a reason if planner annotations dropped non-HF body text; else None.

    Intended HF / page_number removals (OCR sources annotated as skippable) are
    allowed. Any other character-multiset loss is a regression.
    """

    annotations = {item.id: item for item in plan.blocks}
    allowed_drop = {
        _normalize_text_chars(block.text)
        for block in before_blocks
        if block.source == "ocr"
        and (ann := annotations.get(block.id)) is not None
        and ann.role in SKIPPABLE_ROLES
    }
    allowed_drop.discard("")

    before_chars: list[str] = []
    for block in before_blocks:
        norm = _normalize_text_chars(block.text)
        if not norm:
            continue
        if norm in allowed_drop:
            continue
        before_chars.extend(norm)

    after_chars: list[str] = []
    for block in after_blocks:
        norm = _normalize_text_chars(block.text)
        if norm:
            after_chars.extend(norm)

    from collections import Counter

    before_counts = Counter(before_chars)
    after_counts = Counter(after_chars)
    missing = before_counts - after_counts
    if missing:
        sample = "".join(list(missing.elements())[:40])
        return f"dropped {sum(missing.values())} non-HF chars (e.g. {sample!r})"
    return None


def annotate_grouped_blocks_with_plan(
    grouped_blocks: list[TextBlock],
    plan: LayoutPlan,
    *,
    source_blocks: Sequence[TextBlock],
    page_height: float = 540.0,
    skip_header_footer: bool = True,
    font_map: Mapping[str, Mapping[str, str]] | Mapping[str, str] | None = None,
    apply_fonts: bool = False,
) -> list[TextBlock]:
    """Layer planner roles / reading order / HF filter onto heuristic frames.

    Frame geometry and paragraph/list structure stay exactly as ``grouped_blocks``
    (planner-off grouping). Never merges or splits boxes.
    """

    annotations = {item.id: item for item in plan.blocks}
    sources = list(source_blocks)
    resolved_font_map = normalize_font_map(font_map)
    output: list[TextBlock] = []

    for block in grouped_blocks:
        matched = _match_source_blocks(block, sources)
        matched_annotations = [
            annotations[source.id] for source in matched if source.id in annotations
        ]
        if (
            skip_header_footer
            and matched_annotations
            and all(item.role in SKIPPABLE_ROLES for item in matched_annotations)
            and all(source.source == "ocr" for source in matched)
        ):
            continue

        annotated = _copy_block(block)
        if matched_annotations:
            annotated.reading_order = min(item.reading_order for item in matched_annotations)
            all_title = all(item.role in TITLE_ROLES for item in matched_annotations)
            # Only promote to title placeholder when *every* source line is a title.
            # Mixed title+body frames must keep heuristic body structure — otherwise
            # ppt_render's title placeholder writes only the first line and drops body.
            if all_title:
                chosen = max(
                    matched_annotations,
                    key=lambda item: (_ROLE_PRIORITY.get(item.role, 0), item.confidence),
                )
                annotated.block_role = "title"
                annotated.layout_label = "title"
                annotated.use_title_placeholder = True
                font_class = chosen.font_family_class
            else:
                annotated.use_title_placeholder = False
                non_title = [item for item in matched_annotations if item.role not in TITLE_ROLES]
                role_pool = non_title or matched_annotations
                chosen = max(
                    role_pool,
                    key=lambda item: (_ROLE_PRIORITY.get(item.role, 0), item.confidence),
                )
                # Never assign the title role to a mixed frame.
                if chosen.role in TITLE_ROLES:
                    font_class = chosen.font_family_class
                else:
                    annotated.block_role = _ROLE_TO_BLOCK_ROLE.get(chosen.role, annotated.block_role)
                    annotated.layout_label = _ROLE_TO_LAYOUT_LABEL.get(
                        chosen.role, annotated.layout_label
                    )
                    font_class = chosen.font_family_class
                    if chosen.role == "list_item" and annotated.block_role != "list":
                        annotated = _force_list_paragraphs(annotated)
            if apply_fonts:
                annotated.font_family = resolve_font_family(
                    font_class,
                    text=annotated.text,
                    existing_font_family=annotated.font_family,
                    font_map=resolved_font_map,
                    apply_font_class=True,
                )
        output.append(annotated)

    output.sort(key=lambda item: (item.reading_order, item.bbox[1], item.bbox[0]))
    for index, block in enumerate(output):
        block.reading_order = index
    _assign_title_placeholder(output, page_height=page_height)
    # Defense: never leave a multi-line frame on the title-only placeholder.
    for block in output:
        lines = [line for line in (block.text or "").splitlines() if line.strip()]
        if block.use_title_placeholder and len(lines) > 1:
            block.use_title_placeholder = False
    return output


def _match_source_blocks(grouped: TextBlock, sources: Sequence[TextBlock]) -> list[TextBlock]:
    """Associate a heuristic frame with the original line blocks it covers."""

    matched: list[TextBlock] = []
    grouped_text = (grouped.text or "").replace("\n", " ")
    for source in sources:
        source_text = (source.text or "").strip()
        geom = 0.0
        if source.bbox and grouped.bbox:
            geom = intersection_area(source.bbox, grouped.bbox) / max(
                1.0, (source.bbox[2] - source.bbox[0]) * (source.bbox[3] - source.bbox[1])
            )
        text_hit = bool(source_text) and source_text in grouped.text
        # line_bboxes on merged frames often preserve member geometry.
        line_hit = False
        if grouped.line_bboxes:
            for line_bbox in grouped.line_bboxes:
                if (
                    abs(line_bbox[0] - source.bbox[0]) < 2.0
                    and abs(line_bbox[1] - source.bbox[1]) < 2.0
                    and abs(line_bbox[2] - source.bbox[2]) < 4.0
                    and abs(line_bbox[3] - source.bbox[3]) < 4.0
                ):
                    line_hit = True
                    break
        if line_hit or geom >= 0.45 or (text_hit and geom >= 0.15):
            matched.append(source)
            continue
        # Single-line frame identical to a source.
        if source.id == grouped.id or (
            source_text
            and source_text == grouped_text.strip()
            and geom >= 0.25
        ):
            matched.append(source)
    return matched


def _group_is_safe_to_merge(
    group: PlannedGroup,
    members: list[TextBlock],
    *,
    annotations: dict[str, PlannedBlock],
    page_width: float,
    all_blocks: list[TextBlock],
    obstacle_bboxes: Sequence[BBox],
) -> tuple[bool, str]:
    if len(members) <= 1:
        return True, ""
    if group.role in _UNSAFE_GROUP_ROLES:
        return False, f"role {group.role!r} must not merge"
    if any((annotations.get(block.id) and annotations[block.id].role == "table_cell") for block in members):
        return False, "member annotated as table_cell"
    if any(_block_inside_obstacle(block.bbox, obstacle_bboxes) for block in members):
        return False, "member overlaps table/chart region"

    ordered = sorted(members, key=lambda block: (block.bbox[1], block.bbox[0]))
    heights = [max(1.0, block.bbox[3] - block.bbox[1]) for block in ordered]
    median_height = sorted(heights)[len(heights) // 2]
    max_gap = median_height * _MAX_VERTICAL_GAP_RATIO
    max_left_delta = max(4.0, page_width * _MAX_LEFT_DELTA_RATIO)

    font_sizes = [
        block.font_size if block.font_size is not None else _estimate_font_size_from_bbox(block.bbox)
        for block in ordered
    ]
    base_font = sorted(font_sizes)[len(font_sizes) // 2]

    for previous, current in zip(ordered, ordered[1:]):
        gap = current.bbox[1] - previous.bbox[3]
        if gap < -median_height * 0.35:
            return False, "members vertically overlap"
        if gap > max_gap:
            return False, f"vertical gap {gap:.1f} exceeds {max_gap:.1f}"
        left_delta = abs(current.bbox[0] - previous.bbox[0])
        if left_delta > max_left_delta:
            indent_ok = (
                group.role == "list_item"
                and 0 <= (current.bbox[0] - previous.bbox[0]) <= max_left_delta * 2.5
            )
            if not indent_ok:
                return False, f"left edges misaligned ({left_delta:.1f}pt)"
        overlap = max(0.0, min(previous.bbox[2], current.bbox[2]) - max(previous.bbox[0], current.bbox[0]))
        min_width = max(1.0, min(previous.bbox[2] - previous.bbox[0], current.bbox[2] - current.bbox[0]))
        if overlap / min_width < _MIN_COLUMN_OVERLAP_RATIO:
            return False, "members not in the same column"
        curr_font = current.font_size if current.font_size is not None else _estimate_font_size_from_bbox(current.bbox)
        if max(base_font, curr_font) / max(1.0, min(base_font, curr_font)) > _MAX_FONT_SIZE_RATIO:
            return False, "font sizes differ too much"

    union = union_bbox([block.bbox for block in ordered])
    member_ids = {block.id for block in members}
    for other in all_blocks:
        if other.id in member_ids:
            continue
        if _bbox_overlap_ratio(union, other.bbox) >= 0.25:
            return False, f"union overlaps non-member {other.id!r}"
    return True, ""


def _block_inside_obstacle(bbox: BBox, obstacles: Sequence[BBox]) -> bool:
    for obstacle in obstacles:
        if _bbox_overlap_ratio(bbox, obstacle) >= 0.5:
            return True
    return False


def _bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def _annotate_singleton(
    block: TextBlock,
    *,
    annotation: PlannedBlock | None,
    font_map: Mapping[str, Mapping[str, str]],
    apply_fonts: bool,
) -> TextBlock:
    enriched = _enrich_block_paragraphs(_copy_block(block))
    if annotation is None:
        return enriched
    role = annotation.role
    enriched.block_role = _ROLE_TO_BLOCK_ROLE.get(role, "body")
    enriched.layout_label = _ROLE_TO_LAYOUT_LABEL.get(role, "other")
    enriched.reading_order = annotation.reading_order
    # Multi-line blocks must not use the title-only placeholder (drops body lines).
    line_count = len([line for line in (enriched.text or "").splitlines() if line.strip()])
    enriched.use_title_placeholder = role in TITLE_ROLES and line_count <= 1
    if role == "list_item" and not enriched.paragraphs:
        enriched = _force_list_paragraphs(enriched)
    enriched.font_family = resolve_font_family(
        annotation.font_family_class,
        text=enriched.text,
        existing_font_family=enriched.font_family,
        font_map=font_map,
        apply_font_class=apply_fonts,
    )
    # Never pin — keep shrink-to-fit / autofit available.
    enriched.pin_font_size = False
    return enriched


def _merge_members(
    members: list[TextBlock],
    *,
    annotations: dict[str, PlannedBlock],
    role: str,
    reading_order: int,
    group_id: str,
    page_number: int,
    font_map: Mapping[str, Mapping[str, str]],
    apply_fonts: bool,
) -> TextBlock:
    def sort_key(block: TextBlock) -> tuple[float, float]:
        return (block.bbox[1], block.bbox[0])

    ordered = sorted(members, key=sort_key)
    layout_label = _ROLE_TO_LAYOUT_LABEL.get(role, "text")
    paragraphs = _paragraphs_from_members(
        ordered,
        region_label="list" if role == "list_item" else layout_label,  # type: ignore[arg-type]
    )
    text = "\n".join(paragraph.text for paragraph in paragraphs if paragraph.text.strip())
    if not text.strip():
        text = "\n".join(block.text for block in ordered if block.text.strip())

    base = ordered[0]
    bbox = union_bbox([block.bbox for block in ordered])
    font_sizes = [
        block.font_size if block.font_size is not None else _estimate_font_size_from_bbox(block.bbox)
        for block in ordered
    ]
    font_size = sorted(font_sizes)[len(font_sizes) // 2] if font_sizes else base.font_size
    confidence = sum(block.confidence for block in ordered) / max(len(ordered), 1)
    image_boxes = [block.image_bbox for block in ordered if block.image_bbox is not None]
    merged_image_bbox = union_bbox(image_boxes) if image_boxes else base.image_bbox
    line_bboxes = [block.bbox for block in ordered]

    existing_fonts = [block.font_family for block in ordered if is_known_font_family(block.font_family)]
    if existing_fonts:
        font_family: str | None = existing_fonts[0]
    else:
        font_class = "sans"
        best_conf = -1.0
        for block in ordered:
            planned = annotations.get(block.id)
            if planned is not None and planned.confidence >= best_conf:
                font_class = planned.font_family_class
                best_conf = planned.confidence
        font_family = resolve_font_family(
            font_class,
            text=text,
            existing_font_family=None,
            font_map=font_map,
            apply_font_class=apply_fonts,
        )

    line_count = len([line for line in text.splitlines() if line.strip()])
    return TextBlock(
        id=f"plan_text_{page_number}_{group_id}",
        source=base.source,
        bbox=bbox,
        text=text,
        confidence=confidence,
        font_family=font_family,
        font_size=font_size,
        font_color=base.font_color,
        bold=any(block.bold for block in ordered),
        italic=any(block.italic for block in ordered),
        reading_order=reading_order,
        block_role=_ROLE_TO_BLOCK_ROLE.get(role, "body"),
        image_bbox=merged_image_bbox,
        image_polygon=None if image_boxes else base.image_polygon,
        layout_region_id=f"plan_region_{group_id}",
        layout_label=layout_label,
        paragraphs=paragraphs,
        line_bboxes=line_bboxes,
        # Multi-line merges must not use the title-only placeholder (drops body).
        use_title_placeholder=role in TITLE_ROLES and line_count <= 1,
        # Do not pin: shrink-to-fit must remain available (flowchart "Start", etc.).
        pin_font_size=False,
    )


def _force_list_paragraphs(block: TextBlock) -> TextBlock:
    lines = [line for line in block.text.splitlines() if line.strip()] or [block.text]
    paragraphs: list[TextParagraph] = []
    min_left = block.bbox[0]
    for line in lines:
        cleaned, level, list_style, _prefix = strip_bullet_prefix(
            line.strip(), left=block.bbox[0], min_left=min_left
        )
        paragraphs.append(
            TextParagraph(
                text=cleaned,
                list_level=level,
                list_style=list_style,
                bold=block.bold,
                italic=block.italic,
                font_size=block.font_size,
                font_color=block.font_color,
            )
        )
    block.paragraphs = paragraphs
    block.block_role = "list"
    block.layout_label = "list"
    block.text = "\n".join(paragraph.text for paragraph in paragraphs)
    return block


def _copy_block(block: TextBlock) -> TextBlock:
    return TextBlock(
        id=block.id,
        source=block.source,
        bbox=block.bbox,
        text=block.text,
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
        layout_region_id=block.layout_region_id,
        layout_label=block.layout_label,
        paragraphs=list(block.paragraphs) if block.paragraphs else None,
        line_bboxes=list(block.line_bboxes) if block.line_bboxes else None,
        use_title_placeholder=block.use_title_placeholder,
        pin_font_size=block.pin_font_size,
    )


__all__ = [
    "annotate_grouped_blocks_with_plan",
    "apply_layout_plan",
    "planner_text_regression",
]
