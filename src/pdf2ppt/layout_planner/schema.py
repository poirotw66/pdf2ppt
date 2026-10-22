"""JSON Schema for Gemini structured-output layout plans."""

from __future__ import annotations

from typing import Any

from .types import BLOCK_ROLES, FONT_FAMILY_CLASSES

# Gemini structured-output subset of JSON Schema (OpenAPI-ish). Keep enums and
# required fields explicit so invalid plans fail closed in validation.
LAYOUT_PLAN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Must match an existing OCR/geometry block id exactly.",
                    },
                    "role": {
                        "type": "string",
                        "enum": sorted(BLOCK_ROLES),
                    },
                    "reading_order": {
                        "type": "integer",
                        "description": "0-based reading / z-order index.",
                    },
                    "font_family_class": {
                        "type": "string",
                        "enum": sorted(FONT_FAMILY_CLASSES),
                        "description": (
                            "sans/黑體, serif/明體, rounded/圓體, mono, handwriting."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0–1 confidence for this block annotation.",
                    },
                },
                "required": [
                    "id",
                    "role",
                    "reading_order",
                    "font_family_class",
                    "confidence",
                ],
            },
        },
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "member_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Existing block ids to merge into one text frame.",
                    },
                    "role": {
                        "type": "string",
                        "enum": sorted(BLOCK_ROLES),
                    },
                    "reading_order": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "kind": {
                        "type": "string",
                        "enum": ["paragraph", "text_frame"],
                    },
                },
                "required": [
                    "id",
                    "member_ids",
                    "role",
                    "reading_order",
                    "confidence",
                ],
            },
        },
    },
    "required": ["blocks", "groups"],
}


LAYOUT_PLANNER_SYSTEM_PROMPT = """You are a slide layout analyst. Given a page image and a list of
existing OCR/geometry text blocks (id, bbox, text), return STRICT JSON that:
1. Assigns each block a role, reading_order, font_family_class, and confidence.
2. Groups lines into paragraphs/text frames via merge decisions that reference
   EXISTING block ids only (member_ids). Do not invent new text or new ids.
3. Never change OCR text content. Never invent block ids.
4. font_family_class must be one of: sans (黑體), serif (明體), rounded (圓體),
   mono, handwriting.
5. Roles: title, subtitle, body, list_item, caption, header, footer,
   page_number, table_cell, other.
6. Groups must not share member ids. Every grouped member must appear in blocks.
7. Prefer merging vertically stacked lines that form one paragraph; keep
   side-by-side columns as separate groups.
"""


__all__ = ["LAYOUT_PLAN_RESPONSE_SCHEMA", "LAYOUT_PLANNER_SYSTEM_PROMPT"]
