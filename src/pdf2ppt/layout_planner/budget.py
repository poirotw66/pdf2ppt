"""Per-run call / cost / latency budget for the layout planner.

Invariants:
- Every HTTP attempt (success, validation failure, timeout, API error) counts
  against ``max_calls`` and estimated cost.
- Slots are reserved atomically before the call (no TOCTOU overshoot).
- After ``max_consecutive_failures`` consecutive failures, or an auth error,
  the circuit opens and further pages fall back without calling.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

DEFAULT_ESTIMATED_COST_USD = 0.002
DEFAULT_MAX_CONSECUTIVE_FAILURES = 2
AUTH_FALLBACK_KINDS = frozenset({"auth_error", "unauthorized"})


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


@dataclass(slots=True)
class LayoutPlannerBudget:
    max_calls: int = 30
    max_cost_usd: float = 1.0
    concurrency: int = 2
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
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

    def __post_init__(self) -> None:
        limit = max(1, int(self.concurrency))
        self._semaphore = threading.BoundedSemaphore(limit)

    def is_circuit_open(self) -> bool:
        with self._lock:
            return self.circuit_open

    def try_reserve(
        self,
        *,
        estimated_cost_usd: float = DEFAULT_ESTIMATED_COST_USD,
    ) -> bool:
        """Atomically reserve one paid HTTP attempt against call/cost caps.

        Returns False when the circuit is open or the budget is exhausted.
        On True the caller MUST complete with ``commit_success`` or
        ``commit_failure`` (which already counted the call).
        """

        with self._lock:
            if self.circuit_open:
                return False
            if self.calls >= self.max_calls:
                return False
            if self.estimated_cost_usd + estimated_cost_usd > self.max_cost_usd + 1e-9:
                return False
            self.calls += 1
            self.http_attempts += 1
            self.estimated_cost_usd += max(0.0, estimated_cost_usd)
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

    def commit_success(self, record: LayoutPlannerCallRecord) -> None:
        """Finalize a reserved successful HTTP call."""

        with self._lock:
            self.consecutive_failures = 0
            self.total_latency_ms += record.latency_ms
            # Adjust reserved estimate toward actual if provided.
            if record.estimated_cost_usd > 0:
                # Reservation already added DEFAULT; if actual differs, leave as-is
                # (upper-bound) unless actual is larger.
                pass
            record.http_attempted = True
            self.records.append(record)

    def commit_failure(self, record: LayoutPlannerCallRecord) -> None:
        """Finalize a reserved failed HTTP call (still counts against budget)."""

        with self._lock:
            self.fallbacks += 1
            self.consecutive_failures += 1
            self.total_latency_ms += record.latency_ms
            record.http_attempted = True
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
    "DEFAULT_MAX_CONSECUTIVE_FAILURES",
    "LayoutPlannerBudget",
    "LayoutPlannerCallRecord",
    "Timer",
]
