"""Google Gemini / Nano Banana image-edit provider.

Uses the official Gemini Interactions API
(``POST https://generativelanguage.googleapis.com/v1beta/interactions``)
documented for Nano Banana image generation and conversational editing.

Default model: ``gemini-nano-banana-2.1`` (Nano Banana 2.1, GA Oct 2026).

Gemini does not expose an OpenAI-style alpha mask parameter. We:
1. blank the masked pixels on a working copy,
2. pad to the nearest supported aspect ratio and request that ratio,
3. send the blanked image plus an instruction to regenerate only blanked areas,
4. unpad without stretching, then rely on the caller to composite with a hard
   mask so unmasked pixels stay identical to the source.

REST responses carry images under ``steps[*].content[*]`` with
``type: "image"``. The SDK convenience property ``output_image`` is not present
in the raw HTTP payload (see Interactions overview / get-started docs).
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

import numpy as np
from PIL import Image

from ..aspect import (
    pad_image_to_plan,
    pad_rgb_with_edge_mean,
    plan_aspect_pad,
    unpad_image_from_plan,
)
from ..pricing import (
    GEMINI_FALLBACK_COST_USD,
    cost_from_gemini_usage,
    gemini_estimate_cost_usd,
)
from .base import (
    GenerativeEditRequest,
    GenerativeEditResult,
    GenerativeProviderError,
    is_expected_safety_block_message,
)

logger = logging.getLogger(__name__)

GEMINI_DEFAULT_MODEL = "gemini-nano-banana-2.1"
GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
# Alias kept for callers that imported the old flat-rate constant.
GEMINI_DEFAULT_COST_USD = GEMINI_FALLBACK_COST_USD
GEMINI_ENV_API_KEY = "GEMINI_API_KEY"
GEMINI_ENV_API_KEY_ALT = "GOOGLE_API_KEY"
GEMINI_ENV_MODEL = "PDF2PPT_GEMINI_IMAGE_MODEL"


def _image_to_png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _blank_masked_region(image: Image.Image, mask_image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    mask = np.array(mask_image.convert("L"), dtype=np.uint8) > 0
    if mask.shape[:2] != rgb.shape[:2]:
        mask_image = mask_image.convert("L").resize(image.size, Image.Resampling.NEAREST)
        mask = np.array(mask_image, dtype=np.uint8) > 0
    blanked = rgb.copy()
    blanked[mask] = 0
    return Image.fromarray(blanked, mode="RGB")


def _decode_b64_image(payload: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _extract_image_from_interaction(payload: dict[str, Any]) -> Image.Image:
    # SDK convenience property (absent from raw REST payloads).
    output_image = payload.get("output_image")
    if isinstance(output_image, dict):
        data = output_image.get("data")
        if isinstance(data, str) and data:
            return _decode_b64_image(data)

    # Raw REST Interactions responses carry the image under
    # steps[*].content[*] (type "image"); ``output_image`` is an SDK convenience
    # property and is absent from the HTTP payload. Use the last image block.
    steps = payload.get("steps")
    if isinstance(steps, list):
        last_image: str | None = None
        for step in steps:
            if not isinstance(step, dict):
                continue
            content = step.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "image"
                    and isinstance(block.get("data"), str)
                    and block["data"]
                ):
                    last_image = block["data"]
        if last_image:
            return _decode_b64_image(last_image)

    outputs = payload.get("outputs") or payload.get("output")
    if isinstance(outputs, list):
        for item in outputs:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" and isinstance(item.get("data"), str):
                return _decode_b64_image(item["data"])
            inline = item.get("inline_data") or item.get("inlineData")
            if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                return _decode_b64_image(inline["data"])

    # generateContent-shaped fallback.
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inline_data") or part.get("inlineData")
                if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                    return _decode_b64_image(inline["data"])

    raise GenerativeProviderError(
        "Gemini response did not include an output image.",
        may_have_been_billed=True,
        estimated_cost_usd=GEMINI_FALLBACK_COST_USD,
        cost_uncertain=True,
    )


class GeminiImageEditProvider:
    name = "gemini"
    default_model = GEMINI_DEFAULT_MODEL

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = GEMINI_INTERACTIONS_URL,
        timeout_s: float = 120.0,
        cost_usd_per_call: float = GEMINI_DEFAULT_COST_USD,
        transport: Any | None = None,
    ) -> None:
        resolved_key = api_key
        if resolved_key is None:
            resolved_key = os.environ.get(GEMINI_ENV_API_KEY) or os.environ.get(GEMINI_ENV_API_KEY_ALT)
        self._api_key = resolved_key
        env_model = os.environ.get(GEMINI_ENV_MODEL)
        self._model = model or env_model or GEMINI_DEFAULT_MODEL
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
                f"Gemini image edit requires {GEMINI_ENV_API_KEY} or {GEMINI_ENV_API_KEY_ALT}."
            )

        model = request.model or self._model
        source = request.image.convert("RGB")
        blanked = _blank_masked_region(source, request.mask_image)
        plan = plan_aspect_pad(source.width, source.height)
        padded_blanked = pad_rgb_with_edge_mean(blanked, plan)
        # Keep the mask aligned on the padded canvas (blacked regions stay black).
        padded_mask = pad_image_to_plan(request.mask_image.convert("L"), plan, fill=0)
        # Re-blank after padding so pad fill does not reintroduce text into the mask.
        padded_blanked = _blank_masked_region(padded_blanked, padded_mask)

        prompt = (
            f"{request.prompt}\n\n"
            "The attached image has blacked-out regions that must be regenerated. "
            "Preserve every non-blacked pixel's content and geometry as closely as "
            "possible. Return only the completed image."
        )
        body: dict[str, Any] = {
            "model": model,
            "input": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "data": _image_to_png_b64(padded_blanked),
                    "mime_type": "image/png",
                },
            ],
            "response_format": {
                "type": "image",
                "aspect_ratio": plan.aspect_ratio,
                "image_size": "1K",
            },
        }
        response_payload = self._post_json(body)
        image = _extract_image_from_interaction(response_payload)
        image = unpad_image_from_plan(image, plan)

        usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else None
        cost = cost_from_gemini_usage(usage) if isinstance(usage, dict) else None
        if cost is None:
            # Prefer provider override when usage is absent.
            cost = gemini_estimate_cost_usd()
            if self._cost_usd_per_call != GEMINI_FALLBACK_COST_USD:
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
            raw_response_note=f"gemini:/v1beta/interactions aspect={plan.aspect_ratio}",
            cost_source=cost.source,
            cost_uncertain=cost.uncertain,
        )

    def _post_json(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(body=body, api_key=self._api_key)

        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self._base_url,
            data=encoded,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key or "",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"Gemini interactions HTTP {error.code}: {detail[:500]}"
            safety = is_expected_safety_block_message(message)
            # Official copyright/recitation responses state the request is not billed.
            not_charged = "you will not be charged" in detail.lower() or safety
            raise GenerativeProviderError(
                message,
                may_have_been_billed=not not_charged and error.code < 500,
                is_expected_safety_block=safety,
                estimated_cost_usd=0.0 if not_charged else GEMINI_FALLBACK_COST_USD,
                cost_uncertain=not not_charged,
            ) from error
        except urllib.error.URLError as error:
            raise GenerativeProviderError(f"Gemini interactions network error: {error}") from error
        except json.JSONDecodeError as error:
            raise GenerativeProviderError(
                "Gemini interactions returned non-JSON payload.",
                may_have_been_billed=True,
                estimated_cost_usd=GEMINI_FALLBACK_COST_USD,
                cost_uncertain=True,
            ) from error
        if not isinstance(payload, dict):
            raise GenerativeProviderError(
                "Gemini interactions returned an unexpected payload type.",
                may_have_been_billed=True,
                estimated_cost_usd=GEMINI_FALLBACK_COST_USD,
                cost_uncertain=True,
            )
        return payload


__all__ = [
    "GEMINI_DEFAULT_COST_USD",
    "GEMINI_DEFAULT_MODEL",
    "GEMINI_ENV_API_KEY",
    "GEMINI_ENV_API_KEY_ALT",
    "GEMINI_ENV_MODEL",
    "GeminiImageEditProvider",
]
