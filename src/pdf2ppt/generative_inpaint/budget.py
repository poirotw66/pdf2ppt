"""Per-run call/cost budget for generative inpainting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GenerativeCallRecord:
    provider: str
    model: str
    width: int
    height: int
    estimated_cost_usd: float
    cache_hit: bool
    success: bool
    note: str | None = None


@dataclass(slots=True)
class GenerativeBudget:
    max_calls: int = 50
    max_cost_usd: float = 5.0
    calls: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    estimated_cost_usd: float = 0.0
    records: list[GenerativeCallRecord] = field(default_factory=list)

    def can_afford(self, estimated_cost_usd: float) -> bool:
        if self.calls >= self.max_calls:
            return False
        if self.estimated_cost_usd + estimated_cost_usd > self.max_cost_usd + 1e-9:
            return False
        return True

    def record_cache_hit(self, record: GenerativeCallRecord) -> None:
        self.cache_hits += 1
        self.records.append(record)

    def record_call(self, record: GenerativeCallRecord) -> None:
        self.calls += 1
        if record.success:
            self.estimated_cost_usd += record.estimated_cost_usd
        self.records.append(record)

    def record_fallback(self, note: str | None = None) -> None:
        self.fallbacks += 1
        if note:
            self.records.append(
                GenerativeCallRecord(
                    provider="local-fallback",
                    model="opencv-fast",
                    width=0,
                    height=0,
                    estimated_cost_usd=0.0,
                    cache_hit=False,
                    success=True,
                    note=note,
                )
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_calls": self.max_calls,
            "max_cost_usd": self.max_cost_usd,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "fallbacks": self.fallbacks,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "records": [
                {
                    "provider": record.provider,
                    "model": record.model,
                    "width": record.width,
                    "height": record.height,
                    "estimated_cost_usd": record.estimated_cost_usd,
                    "cache_hit": record.cache_hit,
                    "success": record.success,
                    "note": record.note,
                }
                for record in self.records
            ],
        }


__all__ = ["GenerativeBudget", "GenerativeCallRecord"]
