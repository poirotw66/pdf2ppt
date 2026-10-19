"""Cloud generative image-edit providers."""

from __future__ import annotations

from .base import (
    DEFAULT_IMAGE_RESTORE_PROMPT,
    DEFAULT_TEXT_REMOVAL_PROMPT,
    GenerativeEditRequest,
    GenerativeEditResult,
    GenerativeImageEditProvider,
    GenerativeProviderError,
    is_expected_safety_block_message,
)
from .gemini_provider import (
    GEMINI_DEFAULT_MODEL,
    GEMINI_ENV_API_KEY,
    GeminiImageEditProvider,
)
from .openai_provider import (
    OPENAI_DEFAULT_MODEL,
    OPENAI_ENV_API_KEY,
    OpenAIImageEditProvider,
)

PROVIDER_REGISTRY: dict[str, type[OpenAIImageEditProvider] | type[GeminiImageEditProvider]] = {
    "openai": OpenAIImageEditProvider,
    "gemini": GeminiImageEditProvider,
}


def create_provider(
    provider_name: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout_s: float = 120.0,
    transport=None,
) -> OpenAIImageEditProvider | GeminiImageEditProvider:
    normalized = provider_name.strip().lower()
    if normalized not in PROVIDER_REGISTRY:
        known = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unknown generative provider {provider_name!r}. Expected one of: {known}.")
    factory = PROVIDER_REGISTRY[normalized]
    return factory(api_key=api_key, model=model, timeout_s=timeout_s, transport=transport)


__all__ = [
    "DEFAULT_IMAGE_RESTORE_PROMPT",
    "DEFAULT_TEXT_REMOVAL_PROMPT",
    "GEMINI_DEFAULT_MODEL",
    "GEMINI_ENV_API_KEY",
    "OPENAI_DEFAULT_MODEL",
    "OPENAI_ENV_API_KEY",
    "PROVIDER_REGISTRY",
    "GenerativeEditRequest",
    "GenerativeEditResult",
    "GenerativeImageEditProvider",
    "GenerativeProviderError",
    "GeminiImageEditProvider",
    "OpenAIImageEditProvider",
    "create_provider",
    "is_expected_safety_block_message",
]
