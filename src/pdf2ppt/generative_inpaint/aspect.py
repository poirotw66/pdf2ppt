"""Aspect-ratio helpers for cloud image-edit providers.

Gemini Nano Banana picks a discrete output aspect ratio. Stretching that output
back onto a long thin crop distorts the result. We pad the crop to the nearest
supported ratio, request that ratio, then crop the content window back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Documented ImageResponseFormat.aspect_ratio values for the Interactions API.
# Source: https://ai.google.dev/api/interactions-api-v1
GEMINI_SUPPORTED_ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("1:1", 1.0),
    ("2:3", 2.0 / 3.0),
    ("3:2", 3.0 / 2.0),
    ("3:4", 3.0 / 4.0),
    ("4:3", 4.0 / 3.0),
    ("4:5", 4.0 / 5.0),
    ("5:4", 5.0 / 4.0),
    ("9:16", 9.0 / 16.0),
    ("16:9", 16.0 / 9.0),
    ("21:9", 21.0 / 9.0),
    ("1:8", 1.0 / 8.0),
    ("8:1", 8.0 / 1.0),
    ("1:4", 1.0 / 4.0),
    ("4:1", 4.0 / 1.0),
)


@dataclass(frozen=True, slots=True)
class AspectPadPlan:
    aspect_ratio: str
    padded_size: tuple[int, int]  # (width, height)
    content_box: tuple[int, int, int, int]  # left, top, right, bottom in padded coords


def closest_aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "1:1"
    target = width / height
    best_label = GEMINI_SUPPORTED_ASPECT_RATIOS[0][0]
    best_delta = float("inf")
    for label, ratio in GEMINI_SUPPORTED_ASPECT_RATIOS:
        delta = abs(math.log(target / ratio))
        if delta < best_delta:
            best_label, best_delta = label, delta
    return best_label


def plan_aspect_pad(width: int, height: int, *, aspect_ratio: str | None = None) -> AspectPadPlan:
    label = aspect_ratio or closest_aspect_ratio(width, height)
    ratio = next(r for name, r in GEMINI_SUPPORTED_ASPECT_RATIOS if name == label)
    current = width / height if height else 1.0
    if abs(current - ratio) < 1e-6:
        return AspectPadPlan(
            aspect_ratio=label,
            padded_size=(width, height),
            content_box=(0, 0, width, height),
        )
    if current > ratio:
        padded_h = max(height, int(round(width / ratio)))
        padded_w = width
    else:
        padded_w = max(width, int(round(height * ratio)))
        padded_h = height
    left = (padded_w - width) // 2
    top = (padded_h - height) // 2
    return AspectPadPlan(
        aspect_ratio=label,
        padded_size=(padded_w, padded_h),
        content_box=(left, top, left + width, top + height),
    )


def pad_image_to_plan(
    image: Image.Image,
    plan: AspectPadPlan,
    *,
    fill: tuple[int, int, int] | int = (0, 0, 0),
) -> Image.Image:
    mode = image.mode
    canvas = Image.new(mode, plan.padded_size, fill)  # type: ignore[arg-type]
    left, top, _, _ = plan.content_box
    canvas.paste(image, (left, top))
    return canvas


def pad_rgb_with_edge_mean(image: Image.Image, plan: AspectPadPlan) -> Image.Image:
    """Pad with the mean border color so Gemini sees a continuous background."""

    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    if arr.size == 0:
        return pad_image_to_plan(rgb, plan, fill=(0, 0, 0))
    border = np.concatenate(
        [
            arr[0, :, :].reshape(-1, 3),
            arr[-1, :, :].reshape(-1, 3),
            arr[:, 0, :].reshape(-1, 3),
            arr[:, -1, :].reshape(-1, 3),
        ],
        axis=0,
    )
    mean_vals = border.mean(axis=0).round()
    fill = (int(mean_vals[0]), int(mean_vals[1]), int(mean_vals[2]))
    return pad_image_to_plan(rgb, plan, fill=fill)


def unpad_image_from_plan(image: Image.Image, plan: AspectPadPlan) -> Image.Image:
    """Map a model output (any size, same AR) back to the original content window."""

    padded_w, padded_h = plan.padded_size
    left, top, right, bottom = plan.content_box
    content_w = right - left
    content_h = bottom - top
    if image.size == (content_w, content_h):
        return image.convert("RGB")

    # Resize to the padded canvas (same AR => no distortion), then crop content.
    target = image.convert("RGB").resize((padded_w, padded_h), Image.Resampling.LANCZOS)
    cropped = target.crop(plan.content_box)
    if cropped.size != (content_w, content_h):
        cropped = cropped.resize((content_w, content_h), Image.Resampling.LANCZOS)
    return cropped


__all__ = [
    "AspectPadPlan",
    "GEMINI_SUPPORTED_ASPECT_RATIOS",
    "closest_aspect_ratio",
    "pad_image_to_plan",
    "pad_rgb_with_edge_mean",
    "plan_aspect_pad",
    "unpad_image_from_plan",
]
