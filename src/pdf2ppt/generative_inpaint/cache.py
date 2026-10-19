"""Disk cache for generative inpaint patches (image hash + model keyed)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def compute_generative_cache_key(
    *,
    provider: str,
    model: str,
    image: Image.Image,
    mask_image: Image.Image,
    prompt: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(provider.encode("utf-8"))
    digest.update(b"\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(image.convert("RGB").tobytes())
    digest.update(b"\0")
    digest.update(mask_image.convert("L").tobytes())
    return digest.hexdigest()


class GenerativeInpaintCache:
    def __init__(self, cache_dir: Path | None) -> None:
        self._cache_dir = cache_dir
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._cache_dir is not None

    def get(self, key: str) -> Image.Image | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.png"
        if not path.is_file():
            return None
        try:
            with Image.open(path) as handle:
                return handle.convert("RGB").copy()
        except OSError as error:
            logger.warning("Failed to read generative cache entry %s: %s", path, error)
            return None

    def put(self, key: str, image: Image.Image) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{key}.png"
        try:
            image.convert("RGB").save(path, format="PNG")
        except OSError as error:
            logger.warning("Failed to write generative cache entry %s: %s", path, error)


__all__ = ["GenerativeInpaintCache", "compute_generative_cache_key"]
