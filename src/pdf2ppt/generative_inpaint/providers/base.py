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
    """Raised when a provider call fails or returns an unusable payload."""


DEFAULT_TEXT_REMOVAL_PROMPT = (
    "Remove all text, glyphs, logos made of letters, and watermarks from the "
    "masked region. Fill the erased area so it seamlessly continues the "
    "surrounding background, texture, lighting, and perspective. Do not add "
    "any new text, letters, numbers, captions, UI chrome, or watermarks."
)

DEFAULT_IMAGE_RESTORE_PROMPT = (
    "Restore and mildly enhance this image's clarity and compression artifacts "
    "while preserving the original content, composition, colors, and any "
    "existing text exactly. Do not invent new objects or rewrite text."
)


__all__ = [
    "DEFAULT_IMAGE_RESTORE_PROMPT",
    "DEFAULT_TEXT_REMOVAL_PROMPT",
    "GenerativeEditRequest",
    "GenerativeEditResult",
    "GenerativeImageEditProvider",
    "GenerativeProviderError",
]
