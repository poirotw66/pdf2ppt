"""Strict validation of VLM layout plans (fail closed → heuristic fallback)."""

from __future__ import annotations

from typing import Any

from .types import (
    BLOCK_ROLES,
    FONT_FAMILY_CLASSES,
    PROMPT_VERSION,
    LayoutPlan,
    PlannedBlock,
    PlannedGroup,
)

DEFAULT_MIN_CONFIDENCE = 0.55


class LayoutPlanValidationError(ValueError):
    """Raised when a planner response fails schema / consistency checks."""


def parse_and_validate_layout_plan(
    payload: Any,
    *,
    known_block_ids: set[str],
    model: str,
    prompt_version: str = PROMPT_VERSION,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    cache_hit: bool = False,
    latency_ms: float = 0.0,
    estimated_cost_usd: float = 0.0,
) -> LayoutPlan:
    """Parse raw JSON into a validated ``LayoutPlan``.

    Rejects unknown ids, invalid merges, overlapping groups, low confidence,
    and malformed structures. Never invents text.
    """

    if not isinstance(payload, dict):
        raise LayoutPlanValidationError("Layout plan root must be a JSON object.")

    blocks_raw = payload.get("blocks")
    groups_raw = payload.get("groups")
    if not isinstance(blocks_raw, list):
        raise LayoutPlanValidationError("'blocks' must be an array.")
    if not isinstance(groups_raw, list):
        raise LayoutPlanValidationError("'groups' must be an array.")
    if not blocks_raw:
        raise LayoutPlanValidationError("'blocks' must be non-empty.")

    planned_blocks: list[PlannedBlock] = []
    seen_block_ids: set[str] = set()
    for index, item in enumerate(blocks_raw):
        planned_blocks.append(
            _parse_planned_block(
                item,
                index=index,
                known_block_ids=known_block_ids,
                seen_block_ids=seen_block_ids,
                min_confidence=min_confidence,
            )
        )

    # Every known input block should receive an annotation (roles/fonts/order).
    missing = known_block_ids - seen_block_ids
    if missing:
        raise LayoutPlanValidationError(
            f"Plan is missing annotations for block id(s): {sorted(missing)[:8]}"
        )
    extra = seen_block_ids - known_block_ids
    if extra:
        raise LayoutPlanValidationError(
            f"Plan references unknown block id(s): {sorted(extra)[:8]}"
        )

    planned_groups: list[PlannedGroup] = []
    claimed_members: set[str] = set()
    seen_group_ids: set[str] = set()
    for index, item in enumerate(groups_raw):
        group = _parse_planned_group(
            item,
            index=index,
            known_block_ids=known_block_ids,
            claimed_members=claimed_members,
            seen_group_ids=seen_group_ids,
            min_confidence=min_confidence,
        )
        planned_groups.append(group)

    return LayoutPlan(
        blocks=planned_blocks,
        groups=planned_groups,
        model=model,
        prompt_version=prompt_version,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        estimated_cost_usd=estimated_cost_usd,
    )


def _parse_planned_block(
    item: Any,
    *,
    index: int,
    known_block_ids: set[str],
    seen_block_ids: set[str],
    min_confidence: float,
) -> PlannedBlock:
    if not isinstance(item, dict):
        raise LayoutPlanValidationError(f"blocks[{index}] must be an object.")
    block_id = str(item.get("id") or "").strip()
    if not block_id:
        raise LayoutPlanValidationError(f"blocks[{index}].id is required.")
    if block_id not in known_block_ids:
        raise LayoutPlanValidationError(f"blocks[{index}] unknown id {block_id!r}.")
    if block_id in seen_block_ids:
        raise LayoutPlanValidationError(f"Duplicate block annotation for id {block_id!r}.")
    seen_block_ids.add(block_id)

    role = str(item.get("role") or "").strip()
    if role not in BLOCK_ROLES:
        raise LayoutPlanValidationError(f"blocks[{index}].role is invalid: {role!r}.")

    font_class = str(item.get("font_family_class") or "").strip()
    if font_class not in FONT_FAMILY_CLASSES:
        raise LayoutPlanValidationError(
            f"blocks[{index}].font_family_class is invalid: {font_class!r}."
        )

    try:
        reading_order = int(item["reading_order"])
    except (KeyError, TypeError, ValueError) as error:
        raise LayoutPlanValidationError(
            f"blocks[{index}].reading_order must be an integer."
        ) from error

    confidence = _require_confidence(item.get("confidence"), label=f"blocks[{index}].confidence")
    if confidence < min_confidence:
        raise LayoutPlanValidationError(
            f"blocks[{index}] confidence {confidence:.3f} below threshold {min_confidence:.3f}."
        )

    return PlannedBlock(
        id=block_id,
        role=role,  # type: ignore[arg-type]
        reading_order=reading_order,
        font_family_class=font_class,  # type: ignore[arg-type]
        confidence=confidence,
    )


def _parse_planned_group(
    item: Any,
    *,
    index: int,
    known_block_ids: set[str],
    claimed_members: set[str],
    seen_group_ids: set[str],
    min_confidence: float,
) -> PlannedGroup:
    if not isinstance(item, dict):
        raise LayoutPlanValidationError(f"groups[{index}] must be an object.")
    group_id = str(item.get("id") or "").strip()
    if not group_id:
        raise LayoutPlanValidationError(f"groups[{index}].id is required.")
    if group_id in seen_group_ids:
        raise LayoutPlanValidationError(f"Duplicate group id {group_id!r}.")
    seen_group_ids.add(group_id)

    members_raw = item.get("member_ids")
    if not isinstance(members_raw, list) or not members_raw:
        raise LayoutPlanValidationError(f"groups[{index}].member_ids must be a non-empty array.")

    member_ids: list[str] = []
    for member in members_raw:
        member_id = str(member).strip()
        if not member_id:
            raise LayoutPlanValidationError(f"groups[{index}] has an empty member id.")
        if member_id not in known_block_ids:
            raise LayoutPlanValidationError(
                f"groups[{index}] references unknown id {member_id!r}."
            )
        if member_id in claimed_members:
            raise LayoutPlanValidationError(
                f"Contradictory groups: block {member_id!r} appears in multiple groups."
            )
        if member_id in member_ids:
            raise LayoutPlanValidationError(
                f"groups[{index}] repeats member id {member_id!r}."
            )
        claimed_members.add(member_id)
        member_ids.append(member_id)

    role = str(item.get("role") or "").strip()
    if role not in BLOCK_ROLES:
        raise LayoutPlanValidationError(f"groups[{index}].role is invalid: {role!r}.")

    kind = str(item.get("kind") or "text_frame").strip()
    if kind not in {"paragraph", "text_frame"}:
        raise LayoutPlanValidationError(f"groups[{index}].kind is invalid: {kind!r}.")

    try:
        reading_order = int(item["reading_order"])
    except (KeyError, TypeError, ValueError) as error:
        raise LayoutPlanValidationError(
            f"groups[{index}].reading_order must be an integer."
        ) from error

    confidence = _require_confidence(item.get("confidence"), label=f"groups[{index}].confidence")
    if confidence < min_confidence:
        raise LayoutPlanValidationError(
            f"groups[{index}] confidence {confidence:.3f} below threshold {min_confidence:.3f}."
        )

    return PlannedGroup(
        id=group_id,
        member_ids=member_ids,
        role=role,  # type: ignore[arg-type]
        reading_order=reading_order,
        confidence=confidence,
        kind=kind,  # type: ignore[arg-type]
    )


def _require_confidence(value: Any, *, label: str) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as error:
        raise LayoutPlanValidationError(f"{label} must be a number.") from error
    if confidence < 0.0 or confidence > 1.0:
        raise LayoutPlanValidationError(f"{label} must be in [0, 1].")
    return confidence


__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "LayoutPlanValidationError",
    "parse_and_validate_layout_plan",
]
