"""Shared types for the opt-in VLM layout planner (Phase 3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..models import BBox, GroupElement, TextBlock

BlockRole = Literal[
    "title",
    "subtitle",
    "body",
    "list_item",
    "caption",
    "header",
    "footer",
    "page_number",
    "table_cell",
    "other",
]

FontFamilyClass = Literal["sans", "serif", "rounded", "mono", "handwriting"]

BLOCK_ROLES: frozenset[str] = frozenset(
    {
        "title",
        "subtitle",
        "body",
        "list_item",
        "caption",
        "header",
        "footer",
        "page_number",
        "table_cell",
        "other",
    }
)

FONT_FAMILY_CLASSES: frozenset[str] = frozenset(
    {"sans", "serif", "rounded", "mono", "handwriting"}
)

# Roles that map to header/footer filtering when skip_header_footer is on.
SKIPPABLE_ROLES: frozenset[str] = frozenset({"header", "footer", "page_number"})

# Roles that should use the slide title placeholder when near the top.
TITLE_ROLES: frozenset[str] = frozenset({"title"})

PROMPT_VERSION = "layout-planner-v1"


@dataclass(slots=True)
class PlannedBlock:
    """Per-block annotations from the VLM (never changes OCR text)."""

    id: str
    role: BlockRole
    reading_order: int
    font_family_class: FontFamilyClass
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PlannedGroup:
    """Merge/split decision referencing existing block ids only."""

    id: str
    member_ids: list[str]
    role: BlockRole
    reading_order: int
    confidence: float
    kind: Literal["paragraph", "text_frame"] = "text_frame"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LayoutPlan:
    """Validated structured layout plan for one page."""

    blocks: list[PlannedBlock]
    groups: list[PlannedGroup]
    model: str
    prompt_version: str = PROMPT_VERSION
    cache_hit: bool = False
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "groups": [group.to_dict() for group in self.groups],
            "model": self.model,
            "prompt_version": self.prompt_version,
            "cache_hit": self.cache_hit,
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "notes": list(self.notes),
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(slots=True)
class LayoutPlannerApplication:
    """Result of applying a validated plan onto geometric text blocks."""

    text_blocks: list[TextBlock]
    group_elements: list[GroupElement]
    plan_metadata: dict[str, Any]
    used_planner: bool
    fallback_reason: str | None = None
    # When False, caller must keep heuristic paragraph/list grouping and then
    # layer roles via annotate_grouped_blocks_with_plan.
    plan: LayoutPlan | None = None
    frames_applied: bool = True


@dataclass(slots=True)
class BlockEvidence:
    """Compact block payload sent to the VLM (ids + geometry + text)."""

    id: str
    bbox: BBox
    text: str
    source: str = "ocr"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "bbox": list(self.bbox),
            "text": self.text,
            "source": self.source,
        }


__all__ = [
    "BLOCK_ROLES",
    "BlockEvidence",
    "BlockRole",
    "FONT_FAMILY_CLASSES",
    "FontFamilyClass",
    "LayoutPlan",
    "LayoutPlannerApplication",
    "PROMPT_VERSION",
    "PlannedBlock",
    "PlannedGroup",
    "SKIPPABLE_ROLES",
    "TITLE_ROLES",
]
