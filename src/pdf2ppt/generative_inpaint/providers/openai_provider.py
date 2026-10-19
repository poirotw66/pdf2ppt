"""OpenAI Images Edit provider (GPT Image models).

Uses the official ``POST /v1/images/edits`` API. Default model is ``gpt-image-2``
(documented GPT Image edit model that accepts an alpha mask). Callers may
override to newer IDs such as ``gpt-image-2.5-sunburst`` via options/env.

Mask convention (OpenAI docs): fully transparent pixels (alpha = 0) mark the
region to edit. We convert our L-mode text mask accordingly.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from PIL import Image

from ..pricing import (
    OPENAI_FALLBACK_COST_USD,
    cost_from_openai_usage,
    openai_estimate_cost_usd,
)
from .base import (
    GenerativeEditRequest,
    GenerativeEditResult,
    GenerativeProviderError,
    is_expected_safety_block_message,
)

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_MODEL = "gpt-image-2"
OPENAI_IMAGES_EDIT_URL = "https://api.openai.com/v1/images/edits"
OPENAI_DEFAULT_COST_USD = OPENAI_FALLBACK_COST_USD
OPENAI_ENV_API_KEY = "OPENAI_API_KEY"
OPENAI_ENV_MODEL = "PDF2PPT_OPENAI_IMAGE_MODEL"


def _rgba_mask_from_l_mask(mask_image: Image.Image) -> Image.Image:
    """Build an RGBA mask where alpha=0 means 'edit here' (OpenAI convention)."""

    luminance = mask_image.convert("L")
    rgba = Image.new("RGBA", luminance.size, (0, 0, 0, 255))
    alpha = luminance.point(lambda value: 0 if value > 0 else 255)
    rgba.putalpha(alpha)
    return rgba


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_b64_image(payload: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _build_multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = "----pdf2ppt-openai-boundary"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")
    for name, (filename, content, content_type) in files.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
        )
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class OpenAIImageEditProvider:
    name = "openai"
    default_model = OPENAI_DEFAULT_MODEL

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = OPENAI_IMAGES_EDIT_URL,
        timeout_s: float = 120.0,
        cost_usd_per_call: float = OPENAI_DEFAULT_COST_USD,
        transport: Any | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get(OPENAI_ENV_API_KEY)
        env_model = os.environ.get(OPENAI_ENV_MODEL)
        self._model = model or env_model or OPENAI_DEFAULT_MODEL
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._cost_usd_per_call = cost_usd_per_call
        self._transport = transport

    @property
    def configured_model(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def estimate_cost_usd(self, *, model: str, width: int, height: int) -> float:
        del model, width, height
        return self._cost_usd_per_call

    def edit(self, request: GenerativeEditRequest) -> GenerativeEditResult:
        if not self._api_key:
            raise GenerativeProviderError(
                f"OpenAI image edit requires {OPENAI_ENV_API_KEY} (or an explicit api_key)."
            )

        model = request.model or self._model
        source = request.image.convert("RGB")
        mask_rgba = _rgba_mask_from_l_mask(request.mask_image)
        if mask_rgba.size != source.size:
            mask_rgba = mask_rgba.resize(source.size, Image.Resampling.NEAREST)

        fields = {
            "model": model,
            "prompt": request.prompt,
            "n": "1",
            "output_format": "png",
        }
        # gpt-image-2 rejects transparent backgrounds; keep opaque/auto-safe defaults.
        if model.startswith("gpt-image-2") and "gpt-image-2.5" not in model:
            fields["background"] = "opaque"

        body, content_type = _build_multipart(
            fields,
            {
                "image": ("image.png", _image_to_png_bytes(source), "image/png"),
                "mask": ("mask.png", _image_to_png_bytes(mask_rgba), "image/png"),
            },
        )
        response_payload = self._post_json(body, content_type)
        image = self._extract_image(response_payload)
        if image.size != source.size:
            image = image.resize(source.size, Image.Resampling.LANCZOS)

        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        cost = cost_from_openai_usage(usage) if isinstance(usage, dict) else None
        if cost is None:
            cost = openai_estimate_cost_usd()
            if self._cost_usd_per_call != OPENAI_FALLBACK_COST_USD:
                cost = type(cost)(
                    usd=self._cost_usd_per_call,
                    source="estimate",
                    uncertain=True,
                )

        return GenerativeEditResult(
            image=image,
            model=model,
            provider_name=self.name,
            estimated_cost_usd=cost.usd,
            usage=usage if isinstance(usage, dict) else None,
            raw_response_note="openai:/v1/images/edits",
            cost_source=cost.source,
            cost_uncertain=cost.uncertain,
        )

    def _post_json(self, body: bytes, content_type: str) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(body=body, content_type=content_type, api_key=self._api_key)

        request = urllib.request.Request(
            self._base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": content_type,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"OpenAI images.edit HTTP {error.code}: {detail[:500]}"
            safety = is_expected_safety_block_message(message)
            raise GenerativeProviderError(
                message,
                may_have_been_billed=not safety and error.code < 500,
                is_expected_safety_block=safety,
                estimated_cost_usd=0.0 if safety else OPENAI_FALLBACK_COST_USD,
                cost_uncertain=not safety,
            ) from error
        except urllib.error.URLError as error:
            raise GenerativeProviderError(f"OpenAI images.edit network error: {error}") from error
        except json.JSONDecodeError as error:
            raise GenerativeProviderError(
                "OpenAI images.edit returned non-JSON payload.",
                may_have_been_billed=True,
                estimated_cost_usd=OPENAI_FALLBACK_COST_USD,
                cost_uncertain=True,
            ) from error
        if not isinstance(payload, dict):
            raise GenerativeProviderError(
                "OpenAI images.edit returned an unexpected payload type.",
                may_have_been_billed=True,
                estimated_cost_usd=OPENAI_FALLBACK_COST_USD,
                cost_uncertain=True,
            )
        return payload

    def _extract_image(self, payload: dict[str, Any]) -> Image.Image:
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise GenerativeProviderError(
                "OpenAI images.edit response missing data[].",
                may_have_been_billed=True,
                estimated_cost_usd=OPENAI_FALLBACK_COST_USD,
                cost_uncertain=True,
            )
        first = data[0]
        if not isinstance(first, dict):
            raise GenerativeProviderError(
                "OpenAI images.edit data[0] is not an object.",
                may_have_been_billed=True,
                estimated_cost_usd=OPENAI_FALLBACK_COST_USD,
                cost_uncertain=True,
            )
        b64 = first.get("b64_json")
        if not isinstance(b64, str) or not b64:
            raise GenerativeProviderError(
                "OpenAI images.edit response missing b64_json (set response_format/b64).",
                may_have_been_billed=True,
                estimated_cost_usd=OPENAI_FALLBACK_COST_USD,
                cost_uncertain=True,
            )
        return _decode_b64_image(b64)


__all__ = [
    "OPENAI_DEFAULT_COST_USD",
    "OPENAI_DEFAULT_MODEL",
    "OPENAI_ENV_API_KEY",
    "OPENAI_ENV_MODEL",
    "OpenAIImageEditProvider",
]
