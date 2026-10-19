"""Provider contract for generative image-edit backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(slots=True)
class GenerativeEditRequest:
    """One masked-region edit request.

    ``mask_image`` is an L-mode image where non-zero pixels mark the region that
    may change. Providers may convert this into an alpha mask (OpenAI) or a
    conversational edit prompt (Gemini).
    """

    image: Image.Image
    mask_image: Image.Image
    prompt: str
    model: str
    size_hint: str | None = None


@dataclass(slots=True)
class GenerativeEditResult:
    image: Image.Image
    model: str
    provider_name: str
    estimated_cost_usd: float
    usage: dict[str, object] | None = None
    raw_response_note: str | None = None
    cost_source: str = "estimate"  # "usage" | "estimate" | "zero"
    cost_uncertain: bool = False


class GenerativeImageEditProvider(Protocol):
    """Swappable cloud image-edit backend."""

    name: str
    default_model: str

    def is_configured(self) -> bool:
        """True when credentials required for live calls are present."""

    def edit(self, request: GenerativeEditRequest) -> GenerativeEditResult:
        """Edit ``request.image`` guided by ``request.mask_image`` + prompt."""

    def estimate_cost_usd(self, *, model: str, width: int, height: int) -> float:
        """Conservative pre-call cost estimate used for budget gating."""


class GenerativeProviderError(RuntimeError):
    """Raised when a provider call fails or returns an unusable payload.

    ``may_have_been_billed`` is True when the HTTP call likely completed enough
    for the vendor to charge (e.g. response parse failure after HTTP 200).
    ``is_expected_safety_block`` is True for copyright/recitation/policy blocks
    that should be treated as expected local fallbacks, not API outages.
    """

    def __init__(
        self,
        message: str,
        *,
        may_have_been_billed: bool = False,
        is_expected_safety_block: bool = False,
        estimated_cost_usd: float | None = None,
        cost_uncertain: bool = False,
    ) -> None:
        super().__init__(message)
        self.may_have_been_billed = may_have_been_billed
        self.is_expected_safety_block = is_expected_safety_block
        self.estimated_cost_usd = estimated_cost_usd
        self.cost_uncertain = cost_uncertain


# Neutral wording: phrases like "remove logos/watermarks" trigger Gemini safety
# filters. Keep the instruction focused on continuing the slide background.
DEFAULT_TEXT_REMOVAL_PROMPT = (
    "This is a crop of a presentation slide background. Fill the masked "
    "(blacked-out) areas so they seamlessly continue the surrounding "
    "background, texture, gradient, lighting, and perspective. Keep the result "
    "clean: no letters, numbers, captions, or other new elements in the filled areas."
)

DEFAULT_IMAGE_RESTORE_PROMPT = (
    "Restore and mildly enhance this image's clarity and compression artifacts "
    "while preserving the original content, composition, colors, and any "
    "existing text exactly. Do not invent new objects or rewrite text."
)


def is_expected_safety_block_message(message: str) -> bool:
    """True for copyright/recitation and similar content-policy blocks."""

    lowered = message.lower()
    markers = (
        "copyright/recitation",
        "copyright",
        "recitation",
        "safety violation",
        "safety violations",
        "image generation blocked",
        "blocked due to",
        "unable to show the generated image",
        "could not generate the image based on the prompt",
        "you will not be charged for this request",
    )
    return any(marker in lowered for marker in markers)


__all__ = [
    "DEFAULT_IMAGE_RESTORE_PROMPT",
    "DEFAULT_TEXT_REMOVAL_PROMPT",
    "GenerativeEditRequest",
    "GenerativeEditResult",
    "GenerativeImageEditProvider",
    "GenerativeProviderError",
    "is_expected_safety_block_message",
]
