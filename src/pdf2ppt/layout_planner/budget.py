"""Per-run call / cost / latency budget for the layout planner.

Invariants:
- Every HTTP attempt (success, validation failure, timeout, API error) counts
  against ``max_calls`` and estimated cost.
- Slots are reserved atomically before the call (no TOCTOU overshoot).
- Dollar-cap checks use a configurable **worst-case next-call** cost so the
  cap cannot be overshot when real usage exceeds the flat estimate.
- After ``max_consecutive_failures`` consecutive failures, or an auth error,
  the circuit opens and further pages fall back without calling.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

# Fallback flat estimate when usageMetadata is absent.
DEFAULT_ESTIMATED_COST_USD = 0.002
DEFAULT_MAX_CONSECUTIVE_FAILURES = 2
AUTH_FALLBACK_KINDS = frozenset({"auth_error", "unauthorized"})

# Published Gemini Flash-Lite list prices (USD per 1M tokens).
DEFAULT_INPUT_USD_PER_MTOK = 0.25
DEFAULT_OUTPUT_USD_PER_MTOK = 1.50
# Worst-case reservation from max_output_tokens (generationConfig) so one call
# cannot blow past the USD cap after reconcile. Typical pages are ~2–4k in /
# ~0.7–1.6k out; reserve for 8k in + 8192 out ≈ $0.0143.
DEFAULT_WORST_CASE_PROMPT_TOKENS = 8_000
DEFAULT_WORST_CASE_OUTPUT_TOKENS = 8_192
DEFAULT_WORST_CASE_CALL_COST_USD = (
    (DEFAULT_WORST_CASE_PROMPT_TOKENS / 1_000_000.0) * DEFAULT_INPUT_USD_PER_MTOK
    + (DEFAULT_WORST_CASE_OUTPUT_TOKENS / 1_000_000.0) * DEFAULT_OUTPUT_USD_PER_MTOK
)


def worst_case_call_cost_usd(
    *,
    prompt_tokens: int = DEFAULT_WORST_CASE_PROMPT_TOKENS,
    max_output_tokens: int = DEFAULT_WORST_CASE_OUTPUT_TOKENS,
    input_usd_per_mtok: float = DEFAULT_INPUT_USD_PER_MTOK,
    output_usd_per_mtok: float = DEFAULT_OUTPUT_USD_PER_MTOK,
) -> float:
    """USD ceiling for one call from prompt + maxOutputTokens worst case."""

    return (max(0, int(prompt_tokens)) / 1_000_000.0) * float(input_usd_per_mtok) + (
        max(0, int(max_output_tokens)) / 1_000_000.0
    ) * float(output_usd_per_mtok)


def cost_from_usage_metadata(
    usage: dict[str, object] | None,
    *,
    input_usd_per_mtok: float = DEFAULT_INPUT_USD_PER_MTOK,
    output_usd_per_mtok: float = DEFAULT_OUTPUT_USD_PER_MTOK,
) -> float | None:
    """Return USD cost from Gemini ``usageMetadata``, or None if unavailable."""

    if not isinstance(usage, dict):
        return None
    prompt_raw = usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0
    output_raw = usage.get("candidatesTokenCount") or usage.get("candidates_token_count") or 0
    prompt_tokens = float(prompt_raw) if isinstance(prompt_raw, (int, float, str)) else 0.0
    output_tokens = float(output_raw) if isinstance(output_raw, (int, float, str)) else 0.0
    if prompt_tokens <= 0 and output_tokens <= 0:
        return None
    return (prompt_tokens / 1_000_000.0) * input_usd_per_mtok + (
        output_tokens / 1_000_000.0
    ) * output_usd_per_mtok


@dataclass(slots=True)
class LayoutPlannerCallRecord:
    model: str
    page_number: int
    cache_hit: bool
    success: bool
    latency_ms: float
    estimated_cost_usd: float
    note: str | None = None
    # validation | timeout | api_error | auth_error | budget | circuit | none
    fallback_kind: str | None = None
    http_attempted: bool = False
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    reserved_cost_usd: float | None = None


@dataclass(slots=True)
class LayoutPlannerBudget:
    max_calls: int = 30
    max_cost_usd: float = 1.0
    concurrency: int = 2
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    worst_case_call_cost_usd: float = DEFAULT_WORST_CASE_CALL_COST_USD
    calls: int = 0
    http_attempts: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    consecutive_failures: int = 0
    circuit_open: bool = False
    circuit_reason: str | None = None
    records: list[LayoutPlannerCallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _semaphore: threading.BoundedSemaphore | None = field(default=None, repr=False, init=False)
    _pending_reservations: deque[float] = field(default_factory=deque, repr=False, init=False)

    def __post_init__(self) -> None:
        limit = max(1, int(self.concurrency))
        self._semaphore = threading.BoundedSemaphore(limit)
        self._pending_reservations = deque()

    def is_circuit_open(self) -> bool:
        with self._lock:
            return self.circuit_open

    def try_reserve(
        self,
        *,
        estimated_cost_usd: float | None = None,
    ) -> bool:
        """Atomically reserve one paid HTTP attempt against call/cost caps.

        Reserves ``worst_case_call_cost_usd`` (or the provided estimate) so the
        dollar cap cannot be overshot when real usage is higher than average.
        On True the caller MUST complete with ``commit_success`` or
        ``commit_failure`` (which already counted the call).
        """

        reserve_cost = (
            float(estimated_cost_usd)
            if estimated_cost_usd is not None
            else float(self.worst_case_call_cost_usd)
        )
        reserve_cost = max(0.0, reserve_cost)
        with self._lock:
            if self.circuit_open:
                return False
            if self.calls >= self.max_calls:
                return False
            if self.estimated_cost_usd + reserve_cost > self.max_cost_usd + 1e-9:
                return False
            self.calls += 1
            self.http_attempts += 1
            self.estimated_cost_usd += reserve_cost
            self._pending_reservations.append(reserve_cost)
            return True

    def acquire_slot(self, timeout_s: float | None = None) -> bool:
        assert self._semaphore is not None
        if timeout_s is None:
            self._semaphore.acquire()
            return True
        return bool(self._semaphore.acquire(timeout=max(0.0, timeout_s)))

    def release_slot(self) -> None:
        assert self._semaphore is not None
        self._semaphore.release()

    def record_cache_hit(self, record: LayoutPlannerCallRecord) -> None:
        with self._lock:
            self.cache_hits += 1
            self.total_latency_ms += record.latency_ms
            self.records.append(record)

    def _pop_reservation(self) -> float:
        if self._pending_reservations:
            return float(self._pending_reservations.popleft())
        return float(self.worst_case_call_cost_usd)

    def commit_success(self, record: LayoutPlannerCallRecord) -> None:
        """Finalize a reserved successful HTTP call; reconcile to actual cost."""

        with self._lock:
            reserved = self._pop_reservation()
            actual = (
                float(record.estimated_cost_usd)
                if record.estimated_cost_usd and record.estimated_cost_usd > 0
                else reserved
            )
            self.estimated_cost_usd += actual - reserved
            # If actual exceeds remaining budget, still record it (already spent)
            # but open the circuit so no further calls are made.
            if self.estimated_cost_usd > self.max_cost_usd + 1e-9:
                self.circuit_open = True
                self.circuit_reason = (
                    f"cost cap reached after reconcile ({self.estimated_cost_usd:.6f} > "
                    f"{self.max_cost_usd:.6f})"
                )
            self.consecutive_failures = 0
            self.total_latency_ms += record.latency_ms
            record.http_attempted = True
            record.reserved_cost_usd = reserved
            record.estimated_cost_usd = actual
            self.records.append(record)

    def commit_failure(self, record: LayoutPlannerCallRecord) -> None:
        """Finalize a reserved failed HTTP call (still counts against budget)."""

        with self._lock:
            reserved = self._pop_reservation()
            # Failed calls may still be billed; keep the reserved (worst-case)
            # amount unless the record carries a known lower actual.
            actual = (
                float(record.estimated_cost_usd)
                if record.estimated_cost_usd and record.estimated_cost_usd > 0
                else reserved
            )
            self.estimated_cost_usd += actual - reserved
            self.fallbacks += 1
            self.consecutive_failures += 1
            self.total_latency_ms += record.latency_ms
            record.http_attempted = True
            record.reserved_cost_usd = reserved
            record.estimated_cost_usd = actual
            self.records.append(record)
            kind = record.fallback_kind or ""
            if kind in AUTH_FALLBACK_KINDS or "API key" in (record.note or ""):
                self.circuit_open = True
                self.circuit_reason = record.note or kind or "auth_error"
            elif self.consecutive_failures >= max(1, int(self.max_consecutive_failures)):
                self.circuit_open = True
                self.circuit_reason = (
                    f"{self.consecutive_failures} consecutive failures"
                    + (f" ({kind})" if kind else "")
                )

    def record_fallback(self, record: LayoutPlannerCallRecord) -> None:
        """Non-HTTP fallback (budget / circuit / missing image) — no call increment."""

        with self._lock:
            self.fallbacks += 1
            self.total_latency_ms += record.latency_ms
            self.records.append(record)

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "max_calls": self.max_calls,
                "max_cost_usd": self.max_cost_usd,
                "concurrency": self.concurrency,
                "max_consecutive_failures": self.max_consecutive_failures,
                "worst_case_call_cost_usd": self.worst_case_call_cost_usd,
                "calls": self.calls,
                "http_attempts": self.http_attempts,
                "cache_hits": self.cache_hits,
                "fallbacks": self.fallbacks,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                "total_latency_ms": round(self.total_latency_ms, 3),
                "consecutive_failures": self.consecutive_failures,
                "circuit_open": self.circuit_open,
                "circuit_reason": self.circuit_reason,
                "records": [
                    {
                        "model": record.model,
                        "page_number": record.page_number,
                        "cache_hit": record.cache_hit,
                        "success": record.success,
                        "latency_ms": round(record.latency_ms, 3),
                        "estimated_cost_usd": record.estimated_cost_usd,
                        "reserved_cost_usd": record.reserved_cost_usd,
                        "prompt_tokens": record.prompt_tokens,
                        "output_tokens": record.output_tokens,
                        "note": record.note,
                        "fallback_kind": record.fallback_kind,
                        "http_attempted": record.http_attempted,
                    }
                    for record in self.records
                ],
            }


class Timer:
    """Simple wall-clock helper for latency reporting."""

    def __init__(self) -> None:
        self._started = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000.0


__all__ = [
    "AUTH_FALLBACK_KINDS",
    "DEFAULT_ESTIMATED_COST_USD",
    "DEFAULT_INPUT_USD_PER_MTOK",
    "DEFAULT_MAX_CONSECUTIVE_FAILURES",
    "DEFAULT_OUTPUT_USD_PER_MTOK",
    "DEFAULT_WORST_CASE_CALL_COST_USD",
    "DEFAULT_WORST_CASE_OUTPUT_TOKENS",
    "DEFAULT_WORST_CASE_PROMPT_TOKENS",
    "LayoutPlannerBudget",
    "LayoutPlannerCallRecord",
    "Timer",
    "cost_from_usage_metadata",
    "worst_case_call_cost_usd",
]
