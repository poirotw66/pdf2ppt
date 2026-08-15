"""The ``white-box`` background inpainting engine: paints masked regions solid white.

Split out of the former monolithic ``inpainting_engines.py`` (Phase 1.3, pure move).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import BackgroundInpaintingEngine


class WhiteBoxInpaintingEngine(BackgroundInpaintingEngine):
    name = "white-box"

    def inpaint(self, page_image: Image.Image, mask_image: Image.Image) -> Image.Image:
        mask_array = np.array(mask_image.convert("L")) > 0
        result = np.array(page_image.convert("RGB"), copy=True)
        result[mask_array] = 255
        return Image.fromarray(result, mode="RGB")


__all__ = ["WhiteBoxInpaintingEngine"]
