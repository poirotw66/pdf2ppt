"""Per-run call / cost / latency budget for the layout planner."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class LayoutPlannerCallRecord:
    model: str
    page_number: int
    cache_hit: bool
    success: bool
    latency_ms: float
    estimated_cost_usd: float
    note: str | None = None
    fallback_kind: str | None = None  # validation | timeout | api_error | budget | none


@dataclass(slots=True)
class LayoutPlannerBudget:
    max_calls: int = 30
    max_cost_usd: float = 1.0
    concurrency: int = 2
    calls: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    records: list[LayoutPlannerCallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _semaphore: threading.BoundedSemaphore | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        limit = max(1, int(self.concurrency))
        self._semaphore = threading.BoundedSemaphore(limit)

    def can_afford(self, estimated_cost_usd: float = 0.0) -> bool:
        with self._lock:
            if self.calls >= self.max_calls:
                return False
            if self.estimated_cost_usd + estimated_cost_usd > self.max_cost_usd + 1e-9:
                return False
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

    def record_call(self, record: LayoutPlannerCallRecord) -> None:
        with self._lock:
            if not record.cache_hit:
                self.calls += 1
                self.estimated_cost_usd += max(0.0, record.estimated_cost_usd)
            self.total_latency_ms += record.latency_ms
            self.records.append(record)

    def record_fallback(self, record: LayoutPlannerCallRecord) -> None:
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
                "calls": self.calls,
                "cache_hits": self.cache_hits,
                "fallbacks": self.fallbacks,
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                "total_latency_ms": round(self.total_latency_ms, 3),
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


__all__ = ["LayoutPlannerBudget", "LayoutPlannerCallRecord", "Timer"]
