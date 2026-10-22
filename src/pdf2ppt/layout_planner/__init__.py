"""Opt-in VLM layout planner (Phase 3 / v1.5).

Produces structured JSON (roles, reading order, paragraph groups, font class)
validated by schema. Never returns pixels and never invents OCR text. Disabled
by default — without ``--layout-planner`` / API key the geometric heuristic path
is unchanged (byte-identical PPTX).
"""

from __future__ import annotations

from .apply import annotate_grouped_blocks_with_plan, apply_layout_plan
from .base import LayoutPlanner, LayoutPlannerError, blocks_to_evidence
from .budget import LayoutPlannerBudget, LayoutPlannerCallRecord
from .cache import LayoutPlannerCache, compute_layout_planner_cache_key
from .fonts import (
    CJK_FONT_FALLBACKS,
    DEFAULT_FONT_MAP,
    font_script_for_text,
    is_known_font_family,
    normalize_font_map,
    resolve_font_family,
    text_has_cjk,
)
from .gemini import (
    GEMINI_ENV_API_KEY,
    GEMINI_ENV_API_KEY_ALT,
    GEMINI_ENV_LAYOUT_MODEL,
    GEMINI_LAYOUT_DEFAULT_MODEL,
    GeminiLayoutPlanner,
)
from .schema import LAYOUT_PLAN_RESPONSE_SCHEMA, LAYOUT_PLANNER_SYSTEM_PROMPT
from .service import (
    LAYOUT_PLANNER_NAMES,
    LayoutPlannerService,
    build_layout_planner_service,
    create_layout_planner,
    uses_layout_planner,
)
from .types import (
    BLOCK_ROLES,
    FONT_FAMILY_CLASSES,
    PROMPT_VERSION,
    BlockEvidence,
    BlockRole,
    FontFamilyClass,
    LayoutPlan,
    LayoutPlannerApplication,
    PlannedBlock,
    PlannedGroup,
)
from .validate import DEFAULT_MIN_CONFIDENCE, LayoutPlanValidationError, parse_and_validate_layout_plan

__all__ = [
    "BLOCK_ROLES",
    "CJK_FONT_FALLBACKS",
    "DEFAULT_FONT_MAP",
    "DEFAULT_MIN_CONFIDENCE",
    "FONT_FAMILY_CLASSES",
    "annotate_grouped_blocks_with_plan",
    "font_script_for_text",
    "is_known_font_family",
    "text_has_cjk",
    "GEMINI_ENV_API_KEY",
    "GEMINI_ENV_API_KEY_ALT",
    "GEMINI_ENV_LAYOUT_MODEL",
    "GEMINI_LAYOUT_DEFAULT_MODEL",
    "LAYOUT_PLANNER_NAMES",
    "LAYOUT_PLANNER_SYSTEM_PROMPT",
    "LAYOUT_PLAN_RESPONSE_SCHEMA",
    "PROMPT_VERSION",
    "BlockEvidence",
    "BlockRole",
    "FontFamilyClass",
    "GeminiLayoutPlanner",
    "LayoutPlan",
    "LayoutPlanValidationError",
    "LayoutPlanner",
    "LayoutPlannerApplication",
    "LayoutPlannerBudget",
    "LayoutPlannerCache",
    "LayoutPlannerCallRecord",
    "LayoutPlannerError",
    "LayoutPlannerService",
    "PlannedBlock",
    "PlannedGroup",
    "apply_layout_plan",
    "blocks_to_evidence",
    "build_layout_planner_service",
    "compute_layout_planner_cache_key",
    "create_layout_planner",
    "normalize_font_map",
    "parse_and_validate_layout_plan",
    "resolve_font_family",
    "uses_layout_planner",
]
