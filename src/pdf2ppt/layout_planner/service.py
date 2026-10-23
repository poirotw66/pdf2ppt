"""Orchestrate opt-in layout planning with cache, budget, and heuristic fallback."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from ..models import BBox, TextBlock
from .apply import apply_layout_plan
from .base import LayoutPlanner, LayoutPlannerError, blocks_to_evidence
from .budget import (
    DEFAULT_ESTIMATED_COST_USD,
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    DEFAULT_WORST_CASE_CALL_COST_USD,
    LayoutPlannerBudget,
    LayoutPlannerCallRecord,
    Timer,
)
from .cache import LayoutPlannerCache, compute_layout_planner_cache_key
from .gemini import GEMINI_LAYOUT_DEFAULT_MODEL, GeminiLayoutPlanner
from .types import PROMPT_VERSION, LayoutPlannerApplication
from .validate import LayoutPlanValidationError, parse_and_validate_layout_plan

logger = logging.getLogger(__name__)

LAYOUT_PLANNER_NAMES = frozenset({"gemini"})


def uses_layout_planner(name: str | None) -> bool:
    return bool(name) and str(name).strip().lower() in LAYOUT_PLANNER_NAMES


def create_layout_planner(
    name: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 45.0,
    transport: Any | None = None,
    min_confidence: float = 0.55,
) -> LayoutPlanner:
    normalized = str(name).strip().lower()
    if normalized == "gemini":
        return GeminiLayoutPlanner(
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            transport=transport,
            min_confidence=min_confidence,
        )
    raise ValueError(f"Unknown layout planner {name!r}; expected one of {sorted(LAYOUT_PLANNER_NAMES)}.")


class LayoutPlannerService:
    """Page-level planner with disk cache, call budget, and concurrency limit."""

    def __init__(
        self,
        planner: LayoutPlanner,
        *,
        budget: LayoutPlannerBudget | None = None,
        cache_dir: Path | None = None,
        min_confidence: float = 0.55,
        font_map: Mapping[str, Mapping[str, str]] | Mapping[str, str] | None = None,
        skip_header_footer: bool = True,
        apply_grouping: bool = False,
        apply_fonts: bool = False,
    ) -> None:
        self._planner = planner
        self._budget = budget or LayoutPlannerBudget()
        self._cache = LayoutPlannerCache(cache_dir)
        self._min_confidence = min_confidence
        self._font_map = font_map
        self._skip_header_footer = skip_header_footer
        self._apply_grouping = apply_grouping
        self._apply_fonts = apply_fonts

    @property
    def budget(self) -> LayoutPlannerBudget:
        return self._budget

    @property
    def planner_name(self) -> str:
        return getattr(self._planner, "name", "layout_planner")

    @property
    def model(self) -> str:
        return getattr(self._planner, "configured_model", GEMINI_LAYOUT_DEFAULT_MODEL)

    def try_plan_page(
        self,
        *,
        page_image: Image.Image | None,
        text_blocks: list[TextBlock],
        page_number: int,
        page_height: float,
        page_width: float = 960.0,
        table_bboxes: Sequence[BBox] | None = None,
        chart_bboxes: Sequence[BBox] | None = None,
    ) -> LayoutPlannerApplication | None:
        """Return an applied plan, or ``None`` to signal heuristic fallback."""

        if page_image is None or not text_blocks:
            self._budget.record_fallback(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=0.0,
                    estimated_cost_usd=0.0,
                    note="missing page image or text blocks",
                    fallback_kind="validation",
                )
            )
            return None

        evidence = blocks_to_evidence(text_blocks)
        known_ids = {item.id for item in evidence}
        block_payload = [item.to_dict() for item in evidence]
        cache_key = compute_layout_planner_cache_key(
            model=self.model,
            image=page_image,
            block_payload=block_payload,
            prompt_version=PROMPT_VERSION,
        )

        timer = Timer()
        cached = self._cache.get(cache_key)
        if cached is not None:
            try:
                plan = parse_and_validate_layout_plan(
                    cached,
                    known_block_ids=known_ids,
                    model=self.model,
                    prompt_version=PROMPT_VERSION,
                    min_confidence=self._min_confidence,
                    cache_hit=True,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=0.0,
                )
            except LayoutPlanValidationError as error:
                logger.info("Layout planner cache reject page %s: %s", page_number, error)
                self._budget.record_fallback(
                    LayoutPlannerCallRecord(
                        model=self.model,
                        page_number=page_number,
                        cache_hit=True,
                        success=False,
                        latency_ms=timer.elapsed_ms(),
                        estimated_cost_usd=0.0,
                        note=str(error),
                        fallback_kind="validation",
                    )
                )
                return None
            self._budget.record_cache_hit(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=True,
                    success=True,
                    latency_ms=plan.latency_ms,
                    estimated_cost_usd=0.0,
                    note="cache hit",
                    fallback_kind=None,
                )
            )
            return self._apply(
                text_blocks,
                plan,
                page_number=page_number,
                page_height=page_height,
                page_width=page_width,
                table_bboxes=table_bboxes,
                chart_bboxes=chart_bboxes,
            )

        if self._budget.is_circuit_open():
            self._budget.record_fallback(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=0.0,
                    note=self._budget.circuit_reason or "circuit open",
                    fallback_kind="circuit",
                )
            )
            return None

        if not self._planner.is_configured():
            self._budget.record_fallback(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=0.0,
                    note="planner not configured (missing API key)",
                    fallback_kind="api_error",
                )
            )
            return None

        # Atomic reserve before any network I/O. Use worst-case next-call cost
        # so the dollar cap cannot be overshot when usage exceeds the average.
        worst_case = float(
            getattr(self._budget, "worst_case_call_cost_usd", DEFAULT_WORST_CASE_CALL_COST_USD)
        )
        if not self._budget.try_reserve(estimated_cost_usd=worst_case):
            self._budget.record_fallback(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=0.0,
                    note="layout planner budget exhausted",
                    fallback_kind="budget",
                )
            )
            return None

        # Wait for a concurrency slot, but bail out (and refund) if the circuit
        # opens while we are queued — do not dispatch reserved-but-not-started
        # calls after the breaker trips.
        slot_deadline = max(1.0, float(getattr(self._planner, "_timeout_s", 45.0)))
        acquired = False
        waited = 0.0
        poll_s = min(0.25, slot_deadline)
        while waited < slot_deadline + 1e-9:
            if self._budget.is_circuit_open():
                self._budget.cancel_reservation(
                    LayoutPlannerCallRecord(
                        model=self.model,
                        page_number=page_number,
                        cache_hit=False,
                        success=False,
                        latency_ms=timer.elapsed_ms(),
                        estimated_cost_usd=0.0,
                        note=self._budget.circuit_reason or "circuit open before dispatch",
                        fallback_kind="circuit",
                    )
                )
                return None
            step = min(poll_s, max(0.0, slot_deadline - waited))
            if self._budget.acquire_slot(timeout_s=step):
                acquired = True
                break
            waited += step
        if not acquired:
            # Reservation already counted; no HTTP spend — full refund.
            self._budget.cancel_reservation(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=0.0,
                    note="concurrency slot timeout",
                    fallback_kind="timeout",
                )
            )
            return None

        # Re-check after acquiring the slot: another worker may have tripped the
        # breaker while we were waiting.
        if self._budget.is_circuit_open():
            self._budget.release_slot()
            self._budget.cancel_reservation(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=0.0,
                    note=self._budget.circuit_reason or "circuit open before dispatch",
                    fallback_kind="circuit",
                )
            )
            return None

        try:
            plan = self._planner.plan_page(
                page_image=page_image,
                blocks=evidence,
                page_number=page_number,
            )
            plan.latency_ms = timer.elapsed_ms()
            plan.cache_hit = False
        except TimeoutError as error:
            self._budget.commit_failure(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=0.0,
                    note=str(error),
                    fallback_kind="timeout",
                )
            )
            return None
        except (LayoutPlannerError, LayoutPlanValidationError) as error:
            kind = getattr(error, "kind", "validation")
            if isinstance(error, LayoutPlanValidationError):
                kind = "validation"
            # Real usage when the failed response carried usageMetadata; else 0
            # so commit_failure refunds the worst-case reserve (do not pass
            # the reserve itself — that would book it as known spend).
            cost = float(getattr(error, "estimated_cost_usd", 0.0) or 0.0)
            self._budget.commit_failure(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=cost,
                    note=str(error),
                    fallback_kind=str(kind),
                    prompt_tokens=getattr(error, "prompt_tokens", None),
                    output_tokens=getattr(error, "output_tokens", None),
                )
            )
            return None
        except Exception as error:  # noqa: BLE001 — page boundary
            logger.exception("Layout planner unexpected failure on page %s", page_number)
            self._budget.commit_failure(
                LayoutPlannerCallRecord(
                    model=self.model,
                    page_number=page_number,
                    cache_hit=False,
                    success=False,
                    latency_ms=timer.elapsed_ms(),
                    estimated_cost_usd=0.0,
                    note=str(error),
                    fallback_kind="api_error",
                )
            )
            return None
        finally:
            self._budget.release_slot()

        self._cache.put(
            cache_key,
            {
                "blocks": [block.to_dict() for block in plan.blocks],
                "groups": [group.to_dict() for group in plan.groups],
            },
        )
        self._budget.commit_success(
            LayoutPlannerCallRecord(
                model=self.model,
                page_number=page_number,
                cache_hit=False,
                success=True,
                latency_ms=plan.latency_ms,
                estimated_cost_usd=plan.estimated_cost_usd or DEFAULT_ESTIMATED_COST_USD,
                note=None,
                fallback_kind=None,
                prompt_tokens=getattr(plan, "prompt_tokens", None),
                output_tokens=getattr(plan, "output_tokens", None),
            )
        )
        return self._apply(
            text_blocks,
            plan,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
            table_bboxes=table_bboxes,
            chart_bboxes=chart_bboxes,
        )

    def _apply(
        self,
        text_blocks: list[TextBlock],
        plan: Any,
        *,
        page_number: int,
        page_height: float,
        page_width: float,
        table_bboxes: Sequence[BBox] | None,
        chart_bboxes: Sequence[BBox] | None,
    ) -> LayoutPlannerApplication:
        return apply_layout_plan(
            text_blocks,
            plan,
            page_number=page_number,
            page_height=page_height,
            page_width=page_width,
            skip_header_footer=self._skip_header_footer,
            font_map=self._font_map,
            apply_grouping=self._apply_grouping,
            apply_fonts=self._apply_fonts,
            table_bboxes=table_bboxes,
            chart_bboxes=chart_bboxes,
        )


def build_layout_planner_service(options: Any) -> LayoutPlannerService | None:
    """Construct a service from ``ConversionOptions`` when opt-in is set.

    When the flag is set but no API key / transport is available, returns None
    and records a note on ``options.layout_planner_skip_reason`` so the report
    can surface it (byte-identical geometric path).
    """

    name = getattr(options, "layout_planner", None)
    if not uses_layout_planner(name):
        return None
    planner = create_layout_planner(
        str(name),
        model=getattr(options, "layout_planner_model", None),
        api_key=getattr(options, "layout_planner_api_key", None),
        timeout_s=float(getattr(options, "layout_planner_timeout_s", 45.0)),
        transport=getattr(options, "layout_planner_transport", None),
        min_confidence=float(getattr(options, "layout_planner_min_confidence", 0.55)),
    )
    if not planner.is_configured():
        reason = "layout_planner flag set but no API key / transport configured"
        logger.info("%s; using geometric heuristics.", reason)
        try:
            options.layout_planner_skip_reason = reason
            # Still attach an empty budget so the report can show the skip.
            if getattr(options, "layout_planner_budget", None) is None:
                options.layout_planner_budget = LayoutPlannerBudget(
                    max_calls=int(getattr(options, "layout_planner_max_calls", 30)),
                    max_cost_usd=float(getattr(options, "layout_planner_max_cost_usd", 1.0)),
                    concurrency=int(getattr(options, "layout_planner_concurrency", 2)),
                    max_consecutive_failures=int(
                        getattr(
                            options,
                            "layout_planner_max_consecutive_failures",
                            DEFAULT_MAX_CONSECUTIVE_FAILURES,
                        )
                    ),
                )
                options.layout_planner_budget.record_fallback(
                    LayoutPlannerCallRecord(
                        model=getattr(planner, "configured_model", GEMINI_LAYOUT_DEFAULT_MODEL),
                        page_number=0,
                        cache_hit=False,
                        success=False,
                        latency_ms=0.0,
                        estimated_cost_usd=0.0,
                        note=reason,
                        fallback_kind="api_error",
                    )
                )
        except Exception:  # noqa: BLE001
            pass
        return None
    budget = getattr(options, "layout_planner_budget", None)
    if budget is None:
        budget = LayoutPlannerBudget(
            max_calls=int(getattr(options, "layout_planner_max_calls", 30)),
            max_cost_usd=float(getattr(options, "layout_planner_max_cost_usd", 1.0)),
            concurrency=int(getattr(options, "layout_planner_concurrency", 2)),
            max_consecutive_failures=int(
                getattr(
                    options,
                    "layout_planner_max_consecutive_failures",
                    DEFAULT_MAX_CONSECUTIVE_FAILURES,
                )
            ),
        )
        try:
            options.layout_planner_budget = budget
        except Exception:  # noqa: BLE001
            pass
    cache_dir = getattr(options, "layout_planner_cache_dir", None)
    return LayoutPlannerService(
        planner,
        budget=budget,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        min_confidence=float(getattr(options, "layout_planner_min_confidence", 0.55)),
        font_map=getattr(options, "layout_planner_font_map", None),
        skip_header_footer=bool(getattr(options, "layout_skip_header_footer", True)),
        apply_grouping=bool(getattr(options, "layout_planner_apply_grouping", False)),
        apply_fonts=bool(getattr(options, "layout_planner_apply_fonts", False)),
    )


__all__ = [
    "LAYOUT_PLANNER_NAMES",
    "LayoutPlannerService",
    "build_layout_planner_service",
    "create_layout_planner",
    "uses_layout_planner",
]
