"""Map VLM font_family_class labels onto configurable PowerPoint font names."""

from __future__ import annotations

from collections.abc import Mapping

from .types import FontFamilyClass

# Defaults prefer fonts commonly available with the project's measurement stack.
# CJK names use Traditional Chinese labels conceptually (黑體 / 明體 / 圓體).
DEFAULT_FONT_MAP: dict[str, dict[str, str]] = {
    "sans": {"latin": "DejaVu Sans", "cjk": "Noto Sans CJK TC"},
    "serif": {"latin": "DejaVu Serif", "cjk": "Noto Serif CJK TC"},
    "rounded": {"latin": "DejaVu Sans", "cjk": "Noto Sans CJK TC"},
    "mono": {"latin": "DejaVu Sans Mono", "cjk": "Noto Sans Mono CJK TC"},
    "handwriting": {"latin": "DejaVu Sans", "cjk": "Noto Sans CJK TC"},
}


def resolve_font_family(
    font_class: FontFamilyClass | str,
    *,
    script: str,
    font_map: Mapping[str, Mapping[str, str]] | None = None,
) -> str:
    """Return a concrete font family for ``font_class`` given text script."""

    mapping = font_map or DEFAULT_FONT_MAP
    entry = mapping.get(str(font_class)) or DEFAULT_FONT_MAP.get("sans") or {}
    if script in {"cjk", "mixed"}:
        return str(entry.get("cjk") or entry.get("latin") or "Noto Sans CJK TC")
    return str(entry.get("latin") or entry.get("cjk") or "DejaVu Sans")


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
            # Flat single-name maps may use "name" or the value itself as both.
            name = value.get("name")
            if name and not latin and not cjk:
                entry["latin"] = str(name)
                entry["cjk"] = str(name)
            normalized[str(key)] = entry
        else:
            name = str(value)
            normalized[str(key)] = {"latin": name, "cjk": name}
    return normalized


__all__ = ["DEFAULT_FONT_MAP", "normalize_font_map", "resolve_font_family"]
