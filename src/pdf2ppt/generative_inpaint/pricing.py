"""Central pricing config for generative inpaint cost estimates.

Rates are list prices used as budget guardrails, not invoices. Prefer computing
from provider ``usage`` metadata when present; fall back to per-call estimates
when usage is missing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Gemini Nano Banana 2.1 (gemini-nano-banana-2.1) — Gemini Developer API, Standard tier.
# Source: https://ai.google.dev/gemini-api/docs/pricing (verified Oct 2026).
# Measured real cost on ~1.2K input + 1120 image-output + ~300 thinking ≈ $0.038/image.
GEMINI_INPUT_USD_PER_MTOK = 1.50
GEMINI_TEXT_OUTPUT_USD_PER_MTOK = 7.50  # text + thinking
GEMINI_IMAGE_OUTPUT_USD_PER_MTOK = 30.00
# Conservative pre-call guardrail when usage is unknown (typical 1K edit).
GEMINI_FALLBACK_COST_USD = 0.038

# OpenAI GPT Image edits — token rates from the Images API guide (approx.).
# Prefer response usage when present.
OPENAI_TEXT_INPUT_USD_PER_MTOK = 5.00
OPENAI_IMAGE_INPUT_USD_PER_MTOK = 8.00
OPENAI_IMAGE_OUTPUT_USD_PER_MTOK = 30.00
OPENAI_FALLBACK_COST_USD = 0.08


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Resolved USD cost for one provider call."""

    usd: float
    source: str  # "usage" | "estimate" | "zero"
    uncertain: bool


def _as_nonneg_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value == int(value):
        return max(0, int(value))
    return None


def _modality_token_map(entries: object) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(entries, list):
        return result
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        modality = entry.get("modality")
        tokens = _as_nonneg_int(entry.get("tokens"))
        if isinstance(modality, str) and tokens is not None:
            result[modality.lower()] = tokens
    return result


def cost_from_gemini_usage(usage: Mapping[str, Any] | None) -> CostEstimate | None:
    """Compute USD from Interactions API ``usage`` when token counts are present."""

    if not isinstance(usage, Mapping):
        return None

    input_tokens = _as_nonneg_int(usage.get("total_input_tokens"))
    thought_tokens = _as_nonneg_int(usage.get("total_thought_tokens")) or 0
    if input_tokens is None:
        return None

    output_by_modality = _modality_token_map(usage.get("output_tokens_by_modality"))
    image_out = output_by_modality.get("image", 0)
    text_out = output_by_modality.get("text", 0)

    # If modality breakdown is absent, treat remaining output tokens as image
    # (Nano Banana image edits are image-dominated) minus thinking already billed.
    total_output = _as_nonneg_int(usage.get("total_output_tokens"))
    if not output_by_modality and total_output is not None:
        image_out = total_output

    usd = (
        input_tokens * GEMINI_INPUT_USD_PER_MTOK
        + thought_tokens * GEMINI_TEXT_OUTPUT_USD_PER_MTOK
        + text_out * GEMINI_TEXT_OUTPUT_USD_PER_MTOK
        + image_out * GEMINI_IMAGE_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
    return CostEstimate(usd=usd, source="usage", uncertain=False)


def cost_from_openai_usage(usage: Mapping[str, Any] | None) -> CostEstimate | None:
    """Compute USD from OpenAI Images Edit ``usage`` when token counts are present."""

    if not isinstance(usage, Mapping):
        return None

    input_tokens = _as_nonneg_int(usage.get("input_tokens"))
    output_tokens = _as_nonneg_int(usage.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None

    details = usage.get("input_tokens_details")
    text_in = 0
    image_in = 0
    if isinstance(details, Mapping):
        text_in = _as_nonneg_int(details.get("text_tokens")) or 0
        image_in = _as_nonneg_int(details.get("image_tokens")) or 0
    if text_in + image_in == 0:
        # No split: bill all input at the higher image-input rate (conservative).
        image_in = input_tokens
    elif text_in + image_in < input_tokens:
        image_in += input_tokens - (text_in + image_in)

    usd = (
        text_in * OPENAI_TEXT_INPUT_USD_PER_MTOK
        + image_in * OPENAI_IMAGE_INPUT_USD_PER_MTOK
        + output_tokens * OPENAI_IMAGE_OUTPUT_USD_PER_MTOK
    ) / 1_000_000.0
    return CostEstimate(usd=usd, source="usage", uncertain=False)


def gemini_estimate_cost_usd() -> CostEstimate:
    return CostEstimate(usd=GEMINI_FALLBACK_COST_USD, source="estimate", uncertain=True)


def openai_estimate_cost_usd() -> CostEstimate:
    return CostEstimate(usd=OPENAI_FALLBACK_COST_USD, source="estimate", uncertain=True)


def zero_cost(*, source: str = "zero", uncertain: bool = False) -> CostEstimate:
    return CostEstimate(usd=0.0, source=source, uncertain=uncertain)


__all__ = [
    "CostEstimate",
    "GEMINI_FALLBACK_COST_USD",
    "GEMINI_IMAGE_OUTPUT_USD_PER_MTOK",
    "GEMINI_INPUT_USD_PER_MTOK",
    "GEMINI_TEXT_OUTPUT_USD_PER_MTOK",
    "OPENAI_FALLBACK_COST_USD",
    "OPENAI_IMAGE_INPUT_USD_PER_MTOK",
    "OPENAI_IMAGE_OUTPUT_USD_PER_MTOK",
    "OPENAI_TEXT_INPUT_USD_PER_MTOK",
    "cost_from_gemini_usage",
    "cost_from_openai_usage",
    "gemini_estimate_cost_usd",
    "openai_estimate_cost_usd",
    "zero_cost",
]
