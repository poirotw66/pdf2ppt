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


def blocks_to_evidence(blocks: list[Any]) -> list[BlockEvidence]:
    """Convert ``TextBlock``-like objects into compact VLM evidence."""

    evidence: list[BlockEvidence] = []
    for block in blocks:
        evidence.append(
            BlockEvidence(
                id=str(block.id),
                bbox=tuple(float(value) for value in block.bbox[:4]),  # type: ignore[arg-type]
                text=str(block.text),
                source=str(getattr(block, "source", "ocr") or "ocr"),
            )
        )
    return evidence


__all__ = ["LayoutPlanner", "LayoutPlannerError", "blocks_to_evidence"]
