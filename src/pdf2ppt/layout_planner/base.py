"""Pluggable layout-planner interface."""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from .types import BlockEvidence, LayoutPlan


class LayoutPlanner(Protocol):
    """Opt-in page layout planner that returns structured JSON (never pixels)."""

    name: str

    @property
    def configured_model(self) -> str: ...

    def is_configured(self) -> bool:
        """True when credentials / transport are ready for live calls."""

    def plan_page(
        self,
        *,
        page_image: Image.Image,
        blocks: list[BlockEvidence],
        page_number: int = 1,
    ) -> LayoutPlan:
        """Return a validated layout plan or raise on failure.

        Implementations must not mutate OCR text. Callers fall back to geometric
        heuristics on any exception / validation failure / timeout.
        """


class LayoutPlannerError(RuntimeError):
    """Provider / transport failure for a layout planner call."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "api_error",
        estimated_cost_usd: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.estimated_cost_usd = estimated_cost_usd


# Bound prompt size so one page cannot blow past the USD worst-case reserve.
DEFAULT_MAX_EVIDENCE_BLOCKS = 80
DEFAULT_MAX_EVIDENCE_CHARS_PER_BLOCK = 240


def blocks_to_evidence(
    blocks: list[Any],
    *,
    max_blocks: int = DEFAULT_MAX_EVIDENCE_BLOCKS,
    max_chars_per_block: int = DEFAULT_MAX_EVIDENCE_CHARS_PER_BLOCK,
) -> list[BlockEvidence]:
    """Convert ``TextBlock``-like objects into compact VLM evidence.

    Caps block count and per-block text length so a dense page cannot inflate
    the prompt beyond the budgeted worst-case token reserve.
    """

    evidence: list[BlockEvidence] = []
    limit = max(1, int(max_blocks))
    char_limit = max(16, int(max_chars_per_block))
    for block in blocks[:limit]:
        text = str(block.text)
        if len(text) > char_limit:
            text = text[: char_limit - 1] + "…"
        evidence.append(
            BlockEvidence(
                id=str(block.id),
                bbox=tuple(float(value) for value in block.bbox[:4]),  # type: ignore[arg-type]
                text=text,
                source=str(getattr(block, "source", "ocr") or "ocr"),
            )
        )
    return evidence


__all__ = [
    "DEFAULT_MAX_EVIDENCE_BLOCKS",
    "DEFAULT_MAX_EVIDENCE_CHARS_PER_BLOCK",
    "LayoutPlanner",
    "LayoutPlannerError",
    "blocks_to_evidence",
]
