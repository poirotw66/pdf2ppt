"""Apply a validated layout plan onto geometric text blocks / SlideIR groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..layout import (
    _assign_title_placeholder,
    _enrich_block_paragraphs,
    _estimate_font_size_from_bbox,
    _paragraphs_from_members,
    strip_bullet_prefix,
    union_bbox,
)
from ..models import GroupElement, LayoutRegion, TextBlock, TextParagraph
from ..text_style import classify_text_script
from .fonts import normalize_font_map, resolve_font_family
from .types import (
    SKIPPABLE_ROLES,
    TITLE_ROLES,
    LayoutPlan,
    LayoutPlannerApplication,
    PlannedBlock,
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


def apply_layout_plan(
    text_blocks: list[TextBlock],
    plan: LayoutPlan,
    *,
    page_number: int = 1,
    page_height: float = 540.0,
    skip_header_footer: bool = True,
    font_map: Mapping[str, Mapping[str, str]] | Mapping[str, str] | None = None,
) -> LayoutPlannerApplication:
    """Merge/annotate blocks per plan. Never invents text content."""

    by_id = {block.id: block for block in text_blocks}
    annotations = {item.id: item for item in plan.blocks}
    claimed: set[str] = set()
    for group in plan.groups:
        claimed.update(group.member_ids)

    resolved_font_map = normalize_font_map(font_map)
    output_blocks: list[TextBlock] = []
    group_elements: list[GroupElement] = []
    regions: list[LayoutRegion] = []

    for group in sorted(plan.groups, key=lambda item: item.reading_order):
        members = [by_id[member_id] for member_id in group.member_ids if member_id in by_id]
        if not members:
            continue
        if skip_header_footer and group.role in SKIPPABLE_ROLES:
            # Drop OCR HF; keep native so digital pages still show running text.
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
            )
        )

    output_blocks.sort(key=lambda item: (item.reading_order, item.bbox[1], item.bbox[0]))
    for index, block in enumerate(output_blocks):
        block.reading_order = index
    _assign_title_placeholder(output_blocks, page_height=page_height)

    plan_metadata: dict[str, Any] = {
        "engine": "layout_planner",
        "model": plan.model,
        "prompt_version": plan.prompt_version,
        "cache_hit": plan.cache_hit,
        "latency_ms": plan.latency_ms,
        "estimated_cost_usd": plan.estimated_cost_usd,
        "block_count": len(plan.blocks),
        "group_count": len(plan.groups),
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
    )


def _annotate_singleton(
    block: TextBlock,
    *,
    annotation: PlannedBlock | None,
    font_map: Mapping[str, Mapping[str, str]],
) -> TextBlock:
    enriched = _enrich_block_paragraphs(_copy_block(block))
    if annotation is None:
        return enriched
    role = annotation.role
    enriched.block_role = _ROLE_TO_BLOCK_ROLE.get(role, "body")
    enriched.layout_label = _ROLE_TO_LAYOUT_LABEL.get(role, "other")
    enriched.reading_order = annotation.reading_order
    enriched.use_title_placeholder = role in TITLE_ROLES
    if role == "list_item" and not enriched.paragraphs:
        enriched = _force_list_paragraphs(enriched)
    script = classify_text_script(enriched.text)
    enriched.font_family = resolve_font_family(
        annotation.font_family_class,
        script=script,
        font_map=font_map,
    )
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
) -> TextBlock:
    # Preserve planner reading order inside the group when annotations exist.
    def sort_key(block: TextBlock) -> tuple[int, float, float]:
        planned = annotations.get(block.id)
        order = planned.reading_order if planned is not None else 10_000
        return (order, block.bbox[1], block.bbox[0])

    ordered = sorted(members, key=sort_key)
    layout_label = _ROLE_TO_LAYOUT_LABEL.get(role, "text")
    # Reuse list/title paragraph helpers from geometric layout.
    paragraphs = _paragraphs_from_members(
        ordered,
        region_label="list" if role == "list_item" else layout_label,  # type: ignore[arg-type]
    )
    # Never invent text: concatenate existing OCR/native strings only.
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

    # Prefer the highest-confidence font class among members.
    font_class = "sans"
    best_conf = -1.0
    for block in ordered:
        planned = annotations.get(block.id)
        if planned is not None and planned.confidence >= best_conf:
            font_class = planned.font_family_class
            best_conf = planned.confidence
    script = classify_text_script(text)
    font_family = resolve_font_family(font_class, script=script, font_map=font_map)

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
        use_title_placeholder=role in TITLE_ROLES,
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


__all__ = ["apply_layout_plan"]
