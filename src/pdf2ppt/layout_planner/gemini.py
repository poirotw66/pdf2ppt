"""Gemini Flash-Lite layout planner (structured JSON via generateContent)."""

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

from .base import LayoutPlannerError
from .budget import (
    DEFAULT_ESTIMATED_COST_USD,
    DEFAULT_INPUT_USD_PER_MTOK,
    DEFAULT_OUTPUT_USD_PER_MTOK,
    DEFAULT_WORST_CASE_OUTPUT_TOKENS,
    cost_from_usage_metadata,
)
from .schema import LAYOUT_PLAN_RESPONSE_SCHEMA, LAYOUT_PLANNER_SYSTEM_PROMPT
from .types import PROMPT_VERSION, BlockEvidence, LayoutPlan
from .validate import parse_and_validate_layout_plan

logger = logging.getLogger(__name__)

# Lightweight multimodal model with structured-output support (roadmap §3.1).
GEMINI_LAYOUT_DEFAULT_MODEL = "gemini-3.1-flash-lite"
GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_ENV_API_KEY = "GEMINI_API_KEY"
GEMINI_ENV_API_KEY_ALT = "GOOGLE_API_KEY"
GEMINI_ENV_LAYOUT_MODEL = "PDF2PPT_LAYOUT_PLANNER_MODEL"
GEMINI_ENV_INPUT_USD_PER_MTOK = "PDF2PPT_LAYOUT_PLANNER_INPUT_USD_PER_MTOK"
GEMINI_ENV_OUTPUT_USD_PER_MTOK = "PDF2PPT_LAYOUT_PLANNER_OUTPUT_USD_PER_MTOK"
# Keep in sync with budget.DEFAULT_WORST_CASE_OUTPUT_TOKENS (USD reserve).
GEMINI_LAYOUT_MAX_OUTPUT_TOKENS = DEFAULT_WORST_CASE_OUTPUT_TOKENS


def _image_to_jpeg_b64(image: Image.Image, *, quality: int = 85) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _extract_json_text(payload: dict[str, Any]) -> str:
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
            chunks: list[str] = []
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            if chunks:
                return "".join(chunks)
    # Some gateways expose a convenience field.
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text
    raise LayoutPlannerError("Gemini layout response did not include JSON text.", kind="api_error")


def _usage_tokens(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usageMetadata") or payload.get("usage_metadata")
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("promptTokenCount") or usage.get("prompt_token_count")
    output = usage.get("candidatesTokenCount") or usage.get("candidates_token_count")
    try:
        prompt_tokens = int(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_tokens = None
    try:
        output_tokens = int(output) if output is not None else None
    except (TypeError, ValueError):
        output_tokens = None
    return prompt_tokens, output_tokens


def _usage_cost_usd(
    payload: dict[str, Any],
    *,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> float | None:
    usage = payload.get("usageMetadata") or payload.get("usage_metadata")
    if not isinstance(usage, dict):
        return None
    return cost_from_usage_metadata(
        usage,
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
    )


class GeminiLayoutPlanner:
    """Opt-in Gemini layout planner using structured JSON output."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 45.0,
        transport: Any | None = None,
        min_confidence: float = 0.55,
        estimated_cost_usd: float = DEFAULT_ESTIMATED_COST_USD,
        input_usd_per_mtok: float | None = None,
        output_usd_per_mtok: float | None = None,
    ) -> None:
        resolved_key = api_key
        if resolved_key is None:
            resolved_key = os.environ.get(GEMINI_ENV_API_KEY) or os.environ.get(GEMINI_ENV_API_KEY_ALT)
        self._api_key = resolved_key
        env_model = os.environ.get(GEMINI_ENV_LAYOUT_MODEL)
        self._model = model or env_model or GEMINI_LAYOUT_DEFAULT_MODEL
        self._timeout_s = timeout_s
        self._transport = transport
        self._min_confidence = min_confidence
        self._estimated_cost_usd = estimated_cost_usd
        self._input_usd_per_mtok = (
            float(input_usd_per_mtok)
            if input_usd_per_mtok is not None
            else float(os.environ.get(GEMINI_ENV_INPUT_USD_PER_MTOK) or DEFAULT_INPUT_USD_PER_MTOK)
        )
        self._output_usd_per_mtok = (
            float(output_usd_per_mtok)
            if output_usd_per_mtok is not None
            else float(os.environ.get(GEMINI_ENV_OUTPUT_USD_PER_MTOK) or DEFAULT_OUTPUT_USD_PER_MTOK)
        )

    @property
    def configured_model(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return bool(self._api_key) or self._transport is not None

    def plan_page(
        self,
        *,
        page_image: Image.Image,
        blocks: list[BlockEvidence],
        page_number: int = 1,
    ) -> LayoutPlan:
        if not self.is_configured():
            raise LayoutPlannerError(
                f"Gemini layout planner requires {GEMINI_ENV_API_KEY} or {GEMINI_ENV_API_KEY_ALT}.",
                kind="api_error",
            )

        block_payload = [block.to_dict() for block in blocks]
        known_ids = {block.id for block in blocks}
        user_prompt = (
            f"Page {page_number}. Analyze the slide image and these existing text blocks. "
            "Return JSON only matching the schema. Do not invent text or ids.\n\n"
            f"BLOCKS:\n{json.dumps(block_payload, ensure_ascii=False)}"
        )
        body = {
            "systemInstruction": {
                "parts": [{"text": LAYOUT_PLANNER_SYSTEM_PROMPT}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": _image_to_jpeg_b64(page_image),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": LAYOUT_PLAN_RESPONSE_SCHEMA,
                "maxOutputTokens": GEMINI_LAYOUT_MAX_OUTPUT_TOKENS,
            },
        }

        try:
            raw = self._post_generate_content(body)
        except TimeoutError as error:
            raise LayoutPlannerError(str(error), kind="timeout") from error
        except LayoutPlannerError:
            raise
        except Exception as error:  # noqa: BLE001 — transport boundary
            raise LayoutPlannerError(str(error), kind="api_error") from error

        cost = _usage_cost_usd(
            raw,
            input_usd_per_mtok=self._input_usd_per_mtok,
            output_usd_per_mtok=self._output_usd_per_mtok,
        )
        prompt_tokens, output_tokens = _usage_tokens(raw)
        try:
            text = _extract_json_text(raw)
            parsed = json.loads(text)
        except (json.JSONDecodeError, LayoutPlannerError) as error:
            # Bill real token usage when the failed response included it;
            # otherwise leave cost at 0 so the service refunds the reserve.
            raise LayoutPlannerError(
                f"Malformed layout JSON: {error}",
                kind="validation",
                estimated_cost_usd=float(cost) if cost is not None else 0.0,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            ) from error

        plan = parse_and_validate_layout_plan(
            parsed,
            known_block_ids=known_ids,
            model=self._model,
            prompt_version=PROMPT_VERSION,
            min_confidence=self._min_confidence,
            estimated_cost_usd=cost if cost is not None else self._estimated_cost_usd,
        )
        plan.prompt_tokens = prompt_tokens
        plan.output_tokens = output_tokens
        return plan

    def _post_generate_content(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            response = self._transport(body)
            if not isinstance(response, dict):
                raise LayoutPlannerError("Transport returned a non-object response.", kind="api_error")
            return response

        assert self._api_key  # guarded by is_configured
        url = GEMINI_GENERATE_CONTENT_URL.format(model=self._model)
        # Prefer header auth so the key never appears in logs via URL query strings.
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        except TimeoutError as error:
            raise TimeoutError(
                f"Gemini layout planner timed out after {self._timeout_s:.1f}s."
            ) from error
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            # Never include request headers (API key) in the raised message.
            kind = "api_error"
            if error.code in {400, 401, 403}:
                lowered = detail.lower()
                if any(
                    marker in lowered
                    for marker in ("api key", "api_key", "permission", "unauthorized", "forbidden", "not valid")
                ):
                    kind = "auth_error"
            raise LayoutPlannerError(
                f"Gemini layout HTTP {error.code}: {detail}",
                kind=kind,
                estimated_cost_usd=0.0,
            ) from error
        except urllib.error.URLError as error:
            reason = str(error.reason)
            if "timed out" in reason.lower():
                raise TimeoutError(
                    f"Gemini layout planner timed out after {self._timeout_s:.1f}s."
                ) from error
            raise LayoutPlannerError(f"Gemini layout network error: {reason}", kind="api_error") from error

        if not isinstance(payload, dict):
            raise LayoutPlannerError("Gemini layout response was not a JSON object.", kind="api_error")
        return payload


__all__ = [
    "DEFAULT_ESTIMATED_COST_USD",
    "GEMINI_ENV_API_KEY",
    "GEMINI_ENV_API_KEY_ALT",
    "GEMINI_ENV_LAYOUT_MODEL",
    "GEMINI_LAYOUT_DEFAULT_MODEL",
    "GeminiLayoutPlanner",
]
