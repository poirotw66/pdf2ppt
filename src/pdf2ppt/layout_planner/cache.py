"""Disk cache for layout-planner JSON responses (image + prompt + model keyed)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from .types import PROMPT_VERSION

logger = logging.getLogger(__name__)


def compute_layout_planner_cache_key(
    *,
    model: str,
    image: Image.Image,
    block_payload: list[dict[str, Any]],
    prompt_version: str = PROMPT_VERSION,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"layout-planner\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt_version.encode("utf-8"))
    digest.update(b"\0")
    digest.update(image.convert("RGB").tobytes())
    digest.update(b"\0")
    # Stable JSON of block ids/bboxes/text — text is evidence, not inventable.
    digest.update(
        json.dumps(block_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


class LayoutPlannerCache:
    def __init__(self, cache_dir: Path | None) -> None:
        self._cache_dir = cache_dir
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._cache_dir is not None

    def get(self, key: str) -> dict[str, Any] | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Failed to read layout-planner cache entry %s: %s", path, error)
            return None
        return payload if isinstance(payload, dict) else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            logger.warning("Failed to write layout-planner cache entry %s: %s", path, error)


__all__ = ["LayoutPlannerCache", "compute_layout_planner_cache_key"]
