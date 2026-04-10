from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFont

from .models import TextBlock

logger = logging.getLogger(__name__)

DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEFAULT_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
DEFAULT_TEXT_COLOR = "#1F1F1F"


def estimate_font_size(
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    script: str | None = None,
) -> float:
    height = max(1.0, bbox[3] - bbox[1])
    width = max(1.0, bbox[2] - bbox[0])
    script = script or classify_text_script(text)
    script_height_ratio = {
        "cjk": 0.74,
        "latin": 0.78,
        "numeric": 0.82,
        "mixed": 0.76,
        "other": 0.76,
    }.get(script, 0.76)
    base_size = max(6.0, min(72.0, height * script_height_ratio))
    font_path = choose_measurement_font(script)
    if font_path is None:
        logger.debug("No font available for OCR font-size estimation; script=%s", script)
        return base_size

    best_size = base_size
    best_score = float("inf")
    min_size = max(6, int(base_size * 0.65))
    max_size = min(max(min_size + 2, int(base_size * 1.5) + 2), 96)
    target_width = width * script_target_width_ratio(script, text)
    target_height = height * script_target_height_ratio(script)

    candidate_sizes = list(range(min_size, max_size + 1))
    if len(candidate_sizes) <= 12:
        search_sizes = candidate_sizes
    else:
        search_sizes = candidate_sizes[::2]
        if search_sizes[-1] != candidate_sizes[-1]:
            search_sizes.append(candidate_sizes[-1])

    for size in search_sizes:
        score = score_font_size_fit(text, size, font_path, width, height, target_width, target_height)
        if score is None:
            continue
        if score < best_score:
            best_score = score
            best_size = float(size)

    if len(search_sizes) != len(candidate_sizes):
        refine_start = max(min_size, int(best_size) - 1)
        refine_end = min(max_size, int(best_size) + 1)
        for size in range(refine_start, refine_end + 1):
            score = score_font_size_fit(text, size, font_path, width, height, target_width, target_height)
            if score is None:
                continue
            if score < best_score:
                best_score = score
                best_size = float(size)

    return best_size


def score_font_size_fit(
    text: str,
    size: int,
    font_path: str,
    width: float,
    height: float,
    target_width: float,
    target_height: float,
) -> float | None:
    measured_width, measured_height = measure_text_dimensions(text, size, font_path)
    if measured_width <= 0 or measured_height <= 0:
        return None
    width_error = abs(measured_width - target_width) / max(target_width, 1.0)
    height_error = abs(measured_height - target_height) / max(target_height, 1.0)
    overflow_penalty = 0.0
    if measured_width > width * 1.02:
        overflow_penalty += (measured_width - width) / max(width, 1.0)
    if measured_height > height * 1.02:
        overflow_penalty += (measured_height - height) / max(height, 1.0)
    return width_error * 0.7 + height_error * 1.15 + overflow_penalty * 1.5


@lru_cache(maxsize=2048)
def classify_text_script(text: str) -> str:
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return "other"
    cjk_count = sum(1 for char in stripped if is_cjk(char))
    latin_count = sum(1 for char in stripped if char.isascii() and char.isalpha())
    numeric_count = sum(1 for char in stripped if char.isdigit())
    counts = {
        "cjk": cjk_count,
        "latin": latin_count,
        "numeric": numeric_count,
    }
    dominant = max(counts, key=counts.get)
    dominant_count = counts[dominant]
    if dominant_count == 0:
        return "other"
    if dominant_count >= len(stripped) * 0.8:
        return dominant
    return "mixed"


def is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return any(
        (
            0x3400 <= codepoint <= 0x4DBF,
            0x4E00 <= codepoint <= 0x9FFF,
            0xF900 <= codepoint <= 0xFAFF,
        )
    )


def choose_measurement_font(script: str) -> str | None:
    return _choose_measurement_font_cached(script)


@lru_cache(maxsize=8)
def _choose_measurement_font_cached(script: str) -> str | None:
    candidates = [DEFAULT_FONT_PATH]
    if script in {"cjk", "mixed"}:
        candidates = [DEFAULT_CJK_FONT_PATH, DEFAULT_FONT_PATH]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def default_font_family(script: str) -> str:
    if script in {"cjk", "mixed"}:
        return "Noto Sans CJK TC"
    return "DejaVu Sans"


def script_target_width_ratio(script: str, text: str) -> float:
    if script == "numeric":
        return 0.88
    if script == "latin":
        return 0.92
    if script == "cjk":
        return 0.94
    if "\n" in text:
        return 0.96
    return 0.93


def script_target_height_ratio(script: str) -> float:
    if script == "numeric":
        return 0.82
    if script == "latin":
        return 0.84
    if script == "cjk":
        return 0.9
    return 0.87


def ocr_fit_height_cap_ratio(script: str) -> float:
    if script == "numeric":
        return 1.25
    if script == "latin":
        return 1.18
    if script == "cjk":
        return 1.12
    return 1.15


def single_line_fit_width_ratio(script: str) -> float:
    if script == "numeric":
        return 0.92
    if script == "latin":
        return 0.96
    if script == "cjk":
        return 0.94
    return 0.95


@lru_cache(maxsize=512)
def measure_text_dimensions(text: str, font_size: int, font_path: str) -> tuple[float, float]:
    font = load_measurement_font(font_path, font_size)
    lines = text.splitlines() or [text]
    widths: list[float] = []
    heights: list[float] = []
    for line in lines:
        sample = line or " "
        left, top, right, bottom = font.getbbox(sample)
        widths.append(float(right - left))
        heights.append(float(bottom - top))
    line_gap = max(0.0, font_size * 0.15)
    total_height = sum(heights) + max(0, len(lines) - 1) * line_gap
    return max(widths, default=0.0), total_height


@lru_cache(maxsize=128)
def load_measurement_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, font_size)


def gray_image_to_array(gray_crop: Image.Image) -> np.ndarray:
    if gray_crop.mode == "L":
        return np.array(gray_crop, dtype=np.uint8)
    return np.array(gray_crop.convert("L"), dtype=np.uint8)


def extract_text_foreground_mask(gray_crop: Image.Image) -> np.ndarray | None:
    gray = gray_image_to_array(gray_crop)
    if gray.size == 0:
        return None
    if float(np.std(gray)) < 6.0:
        return None

    samples = gray.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _compactness, labels, centers = cv2.kmeans(
        samples,
        2,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    label_map = labels.reshape(gray.shape)
    counts = np.bincount(labels.flatten(), minlength=2)
    foreground_index = int(np.argmin(counts))
    if counts[foreground_index] < max(4, int(gray.size * 0.01)):
        foreground_index = int(np.argmax(np.abs(centers.flatten() - np.mean(gray))))
    return label_map == foreground_index


def format_hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def estimate_text_style(
    text: str,
    color_crop: Image.Image,
    gray_crop: Image.Image,
    *,
    script: str | None = None,
) -> tuple[str, bool]:
    if color_crop.width == 0 or color_crop.height == 0:
        return DEFAULT_TEXT_COLOR, False

    color = np.array(color_crop.convert("RGB"), dtype=np.uint8)
    if color.size == 0:
        return DEFAULT_TEXT_COLOR, False

    foreground_mask = extract_text_foreground_mask(gray_crop)
    return (
        estimate_text_color_from_mask(color, foreground_mask),
        estimate_text_bold_from_mask(text, foreground_mask, script=script),
    )


def estimate_text_color_from_mask(
    color: np.ndarray,
    foreground_mask: np.ndarray | None,
) -> str:
    if foreground_mask is None:
        mean_rgb = tuple(int(round(channel)) for channel in np.mean(color.reshape(-1, 3), axis=0))
        return format_hex_color(mean_rgb)

    foreground_pixels = color[foreground_mask]
    if foreground_pixels.size == 0:
        foreground_pixels = color.reshape(-1, 3)

    dominant_rgb = np.median(foreground_pixels, axis=0)
    rgb = tuple(int(np.clip(round(channel), 0, 255)) for channel in dominant_rgb[:3])
    return format_hex_color(rgb)


def estimate_text_color(color_crop: Image.Image, gray_crop: Image.Image) -> str:
    if color_crop.width == 0 or color_crop.height == 0:
        return DEFAULT_TEXT_COLOR
    color = np.array(color_crop.convert("RGB"), dtype=np.uint8)
    if color.size == 0:
        return DEFAULT_TEXT_COLOR

    foreground_mask = extract_text_foreground_mask(gray_crop)
    return estimate_text_color_from_mask(color, foreground_mask)


def estimate_text_bold(
    text: str,
    gray_crop: Image.Image,
    *,
    script: str | None = None,
) -> bool:
    foreground_mask = extract_text_foreground_mask(gray_crop)
    return estimate_text_bold_from_mask(text, foreground_mask, script=script)


def estimate_text_bold_from_mask(
    text: str,
    foreground_mask: np.ndarray | None,
    *,
    script: str | None = None,
) -> bool:
    if foreground_mask is None:
        return False

    foreground_pixels = int(np.count_nonzero(foreground_mask))
    total_pixels = int(foreground_mask.size)
    if foreground_pixels < max(12, int(total_pixels * 0.02)):
        return False

    ys, xs = np.where(foreground_mask)
    if len(xs) == 0 or len(ys) == 0:
        return False

    bbox_width = max(1, int(xs.max() - xs.min() + 1))
    bbox_height = max(1, int(ys.max() - ys.min() + 1))
    bbox_area = bbox_width * bbox_height
    fill_ratio = foreground_pixels / max(1, bbox_area)

    mask_uint8 = foreground_mask.astype(np.uint8) * 255
    distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 3)
    mean_stroke = float(distance[foreground_mask].mean()) if foreground_pixels else 0.0
    normalized_stroke = mean_stroke / max(1.0, bbox_height)

    script = script or classify_text_script(text)
    fill_threshold = {
        "latin": 0.3,
        "numeric": 0.34,
        "cjk": 0.38,
        "mixed": 0.34,
        "other": 0.34,
    }.get(script, 0.34)
    stroke_threshold = {
        "latin": 0.07,
        "numeric": 0.075,
        "cjk": 0.085,
        "mixed": 0.078,
        "other": 0.078,
    }.get(script, 0.078)
    return fill_ratio >= fill_threshold and normalized_stroke >= stroke_threshold


def build_text_fit_debug_entry(block: TextBlock) -> dict[str, object]:
    script = classify_text_script(block.text)
    font_path = choose_measurement_font(script)
    font_size = max(6.0, block.font_size or 12.0)
    if font_path is None:
        estimated_width = 0.0
        estimated_height = 0.0
    else:
        estimated_width, estimated_height = measure_text_dimensions(
            block.text,
            max(1, int(round(font_size))),
            font_path,
        )

    target_width = max(1.0, block.bbox[2] - block.bbox[0])
    target_height = max(1.0, block.bbox[3] - block.bbox[1])
    width_error = estimated_width - target_width
    height_error = estimated_height - target_height

    return {
        "id": block.id,
        "text": block.text,
        "script": script,
        "font_size_pt": round(font_size, 2),
        "font_path": font_path,
        "target_bbox_pt": {
            "width": round(target_width, 2),
            "height": round(target_height, 2),
        },
        "estimated_ppt_text_pt": {
            "width": round(estimated_width, 2),
            "height": round(estimated_height, 2),
        },
        "width_error_pt": round(width_error, 2),
        "height_error_pt": round(height_error, 2),
        "width_error_ratio": round(width_error / target_width, 4),
        "height_error_ratio": round(height_error / target_height, 4),
    }


def write_text_fit_debug_report(
    *,
    debug_dir: Path,
    page_number: int,
    text_blocks: list[TextBlock],
) -> None:
    entries = [build_text_fit_debug_entry(block) for block in text_blocks if block.source == "ocr"]
    if not entries:
        return

    width_abs_ratios = [abs(entry["width_error_ratio"]) for entry in entries]
    height_abs_ratios = [abs(entry["height_error_ratio"]) for entry in entries]
    payload = {
        "page": page_number,
        "block_count": len(entries),
        "summary": {
            "mean_abs_width_error_ratio": round(sum(width_abs_ratios) / len(width_abs_ratios), 4),
            "mean_abs_height_error_ratio": round(sum(height_abs_ratios) / len(height_abs_ratios), 4),
            "max_abs_width_error_ratio": round(max(width_abs_ratios), 4),
            "max_abs_height_error_ratio": round(max(height_abs_ratios), 4),
        },
        "blocks": entries,
    }
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / f"page_{page_number:03d}_text_fit.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
