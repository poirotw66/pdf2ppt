"""Map VLM font_family_class labels onto configurable PowerPoint font names.

Rules (Phase-3 quality fix):
- Prefer the PDF/OCR block's existing ``font_family`` when it is a real name.
- Only apply the VLM font class when the source font is unknown.
- Use a CJK font only when the text actually contains CJK characters (never for
  Latin+digit "mixed" script).
- Defaults target widely available desktop fonts (not DejaVu/Noto-only).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from ..text_style import is_cjk
from .types import FontFamilyClass

# Cross-platform defaults: Windows / Office themes ship these; macOS maps them
# reasonably. Avoid DejaVu (absent on typical Windows/Mac installs).
DEFAULT_FONT_MAP: dict[str, dict[str, str]] = {
    "sans": {"latin": "Arial", "cjk": "Microsoft JhengHei"},
    "serif": {"latin": "Times New Roman", "cjk": "PMingLiU"},
    "rounded": {"latin": "Arial", "cjk": "Microsoft JhengHei"},
    "mono": {"latin": "Courier New", "cjk": "Microsoft JhengHei"},
    "handwriting": {"latin": "Arial", "cjk": "Microsoft JhengHei"},
}

# Generic / placeholder names that should be replaced by the VLM class map.
_UNKNOWN_FONT_NAMES: frozenset[str] = frozenset(
    {
        "",
        "unknown",
        "default",
        "sans",
        "serif",
        "mono",
        "sans-serif",
        "monospace",
        "null",
        "none",
    }
)


def text_has_cjk(text: str) -> bool:
    return any(is_cjk(char) for char in text)


def font_script_for_text(text: str) -> str:
    """Script used only for font selection: ``cjk`` iff text contains CJK glyphs."""

    return "cjk" if text_has_cjk(text) else "latin"


def is_known_font_family(font_family: str | None) -> bool:
    if font_family is None:
        return False
    cleaned = font_family.strip()
    if not cleaned:
        return False
    if cleaned.lower() in _UNKNOWN_FONT_NAMES:
        return False
    # Bare numeric / punctuation names are not real fonts.
    if re.fullmatch(r"[\d\W_]+", cleaned):
        return False
    return True


def resolve_font_family(
    font_class: FontFamilyClass | str,
    *,
    text: str,
    existing_font_family: str | None = None,
    font_map: Mapping[str, Mapping[str, str]] | None = None,
    apply_font_class: bool = True,
) -> str | None:
    """Resolve a PowerPoint font name without forcing reflow on digital pages.

    When ``existing_font_family`` is a real PDF font, it is kept unchanged.
    When ``apply_font_class`` is False, returns the existing family (or None).
    """

    if is_known_font_family(existing_font_family):
        return existing_font_family
    if not apply_font_class:
        return existing_font_family
    mapping = font_map or DEFAULT_FONT_MAP
    entry = mapping.get(str(font_class)) or DEFAULT_FONT_MAP.get("sans") or {}
    if text_has_cjk(text):
        return str(entry.get("cjk") or entry.get("latin") or "Microsoft JhengHei")
    return str(entry.get("latin") or entry.get("cjk") or "Arial")


def normalize_font_map(
    raw: Mapping[str, Mapping[str, str]] | Mapping[str, str] | None,
) -> dict[str, dict[str, str]]:
    """Accept nested {class: {latin,cjk}} or flat {class: font_name} maps."""

    if not raw:
        return {key: dict(value) for key, value in DEFAULT_FONT_MAP.items()}
    normalized: dict[str, dict[str, str]] = {
        key: dict(value) for key, value in DEFAULT_FONT_MAP.items()
    }
    for key, value in raw.items():
        if isinstance(value, Mapping):
            latin = value.get("latin")
            cjk = value.get("cjk")
            entry = dict(normalized.get(str(key), DEFAULT_FONT_MAP.get("sans", {})))
            if latin:
                entry["latin"] = str(latin)
            if cjk:
                entry["cjk"] = str(cjk)
            name = value.get("name")
            if name and not latin and not cjk:
                entry["latin"] = str(name)
                entry["cjk"] = str(name)
            normalized[str(key)] = entry
        else:
            name = str(value)
            normalized[str(key)] = {"latin": name, "cjk": name}
    return normalized


__all__ = [
    "DEFAULT_FONT_MAP",
    "font_script_for_text",
    "is_known_font_family",
    "normalize_font_map",
    "resolve_font_family",
    "text_has_cjk",
]
