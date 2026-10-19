"""Per-run call/cost budget for generative inpainting."""

from __future__ import annotations

from dataclasses import dataclass, field

# Headroom for free safety-block retries / unbilled attempts beyond billed budget.
DEFAULT_MAX_ATTEMPTS_MULTIPLIER = 3


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
    cost_source: str = "estimate"  # "usage" | "estimate" | "zero"
    cost_uncertain: bool = False
    # None | "api_error" | "safety_block" | "budget" | "ocr_safety" |
    # "cache_reject" | "low_freq" | "flat_ring"
    fallback_kind: str | None = None


@dataclass(slots=True)
class GenerativeBudget:
    max_calls: int = 50
    max_cost_usd: float = 5.0
    # Separate cap on total provider attempts (including free safety blocks).
    # ``None`` → ``max_calls * DEFAULT_MAX_ATTEMPTS_MULTIPLIER``.
    max_attempts: int | None = None
    # ``calls`` is the billed-call counter (backward-compatible name used in reports).
    calls: int = 0
    billed_calls: int = 0
    attempts: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    safety_block_fallbacks: int = 0
    api_error_fallbacks: int = 0
    low_freq_fallbacks: int = 0
    flat_ring_skips: int = 0
    safety_block_retries: int = 0
    estimated_cost_usd: float = 0.0
    cost_uncertain: bool = False
    # Set by the engine so reports show which ring-texture measure drove routing.
    ring_texture_measure: str | None = None
    records: list[GenerativeCallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_attempts is None:
            self.max_attempts = max(1, int(self.max_calls) * DEFAULT_MAX_ATTEMPTS_MULTIPLIER)

    @property
    def resolved_max_attempts(self) -> int:
        if self.max_attempts is None:
            return max(1, int(self.max_calls) * DEFAULT_MAX_ATTEMPTS_MULTIPLIER)
        return int(self.max_attempts)

    def can_afford(self, estimated_cost_usd: float) -> bool:
        # Budget only billed calls against max_calls; free safety blocks do not
        # consume the paid quota (but still count toward max_attempts).
        if self.billed_calls >= self.max_calls or self.calls >= self.max_calls:
            return False
        if self.attempts >= self.resolved_max_attempts:
            return False
        if self.estimated_cost_usd + estimated_cost_usd > self.max_cost_usd + 1e-9:
            return False
        return True

    def record_cache_hit(self, record: GenerativeCallRecord) -> None:
        self.cache_hits += 1
        self.records.append(record)

    def record_call(self, record: GenerativeCallRecord) -> None:
        self.attempts += 1
        # Unbilled safety blocks / free failures must not consume max_calls.
        is_billed = record.estimated_cost_usd > 0
        if is_billed:
            self.billed_calls += 1
            self.calls += 1
            self.estimated_cost_usd += record.estimated_cost_usd
        if record.cost_uncertain:
            self.cost_uncertain = True
        self.records.append(record)

    def record_safety_block_retry(self, note: str | None = None) -> None:
        """Count a free safety-block retry (no fallback yet; another attempt follows)."""

        self.safety_block_retries += 1
        if note:
            self.records.append(
                GenerativeCallRecord(
                    provider="retry",
                    model="safety-block",
                    width=0,
                    height=0,
                    estimated_cost_usd=0.0,
                    cache_hit=False,
                    success=False,
                    note=note,
                    cost_source="zero",
                    cost_uncertain=False,
                    fallback_kind="safety_block_retry",
                )
            )

    def record_fallback(self, note: str | None = None, *, kind: str | None = None) -> None:
        self.fallbacks += 1
        resolved_kind = kind
        if resolved_kind is None and note:
            lowered = note.lower()
            if "flat" in lowered and "ring" in lowered:
                resolved_kind = "flat_ring"
            elif "low-frequency" in lowered or "low_freq" in lowered:
                resolved_kind = "low_freq"
            elif any(
                marker in lowered
                for marker in (
                    "copyright",
                    "recitation",
                    "safety",
                    "blocked",
                    "unable to show the generated image",
                )
            ):
                resolved_kind = "safety_block"
            elif "budget" in lowered:
                resolved_kind = "budget"
            elif "ocr" in lowered:
                resolved_kind = "ocr_safety"
            elif "cache" in lowered:
                resolved_kind = "cache_reject"
            else:
                resolved_kind = "api_error"
        if resolved_kind == "safety_block":
            self.safety_block_fallbacks += 1
        elif resolved_kind == "api_error":
            self.api_error_fallbacks += 1
        elif resolved_kind == "low_freq":
            self.low_freq_fallbacks += 1
        elif resolved_kind == "flat_ring":
            self.flat_ring_skips += 1
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
                    cost_source="zero",
                    cost_uncertain=False,
                    fallback_kind=resolved_kind,
                )
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "max_calls": self.max_calls,
            "max_attempts": self.resolved_max_attempts,
            "max_cost_usd": self.max_cost_usd,
            "calls": self.calls,
            "billed_calls": self.billed_calls,
            "attempts": self.attempts,
            "cache_hits": self.cache_hits,
            "fallbacks": self.fallbacks,
            "safety_block_fallbacks": self.safety_block_fallbacks,
            "api_error_fallbacks": self.api_error_fallbacks,
            "low_freq_fallbacks": self.low_freq_fallbacks,
            "flat_ring_skips": self.flat_ring_skips,
            "ring_texture_measure": self.ring_texture_measure,
            "safety_block_retries": self.safety_block_retries,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "cost_uncertain": self.cost_uncertain,
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
                    "cost_source": record.cost_source,
                    "cost_uncertain": record.cost_uncertain,
                    "fallback_kind": record.fallback_kind,
                }
                for record in self.records
            ],
        }


__all__ = ["DEFAULT_MAX_ATTEMPTS_MULTIPLIER", "GenerativeBudget", "GenerativeCallRecord"]
