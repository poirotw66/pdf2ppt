"""Generate the small, deterministic PDF fixtures checked into tests/fixtures/.

This script is committed alongside the PDFs it produces so the corpus can be
regenerated or extended without needing to source (or re-source) any external
files. Every fixture is programmatically built with PyMuPDF (``fitz``):
backgrounds are either drawn with vector primitives or synthesised as small
raster images (gradients / noise) with a fixed random seed, and all overlaid
text uses PyMuPDF's built-in Base-14 fonts so no font files need to ship with
the repository.

Run it from the repository root:

    python tests/fixtures/generate_fixtures.py

It (re)writes every ``*.pdf`` file in this directory. Nothing here is loaded
by the test suite at import time -- fixtures are consumed as plain files by
``scripts/bench_inpaint.py`` and by tests that open them directly.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

FIXTURES_DIR = Path(__file__).resolve().parent

# 16:9 slide canvas in PDF points (13.333in x 7.5in @ 72dpi), matching the
# aspect ratio pdf2ppt targets for PPTX output.
PAGE_WIDTH = 960.0
PAGE_HEIGHT = 540.0
PAGE_RECT = fitz.Rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT)
PAGE_AREA = PAGE_WIDTH * PAGE_HEIGHT

LOREM = (
    "The quick brown fox jumps over the lazy dog while the committee reviews "
    "quarterly targets and revises the roadmap for the next fiscal cycle. "
    "Stakeholders expect a concise summary before the offsite meeting begins."
)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _jpeg_bytes(image: Image.Image, quality: int = 78) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _new_doc() -> tuple[fitz.Document, fitz.Page]:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    return doc, page


def _save(doc: fitz.Document, name: str) -> None:
    path = FIXTURES_DIR / name
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    size_kb = path.stat().st_size / 1024.0
    print(f"wrote {path.name:32s} {size_kb:8.1f} KiB")


def _insert_background_image(page: fitz.Page, image: Image.Image, *, image_format: str = "png") -> None:
    stream = _jpeg_bytes(image) if image_format == "jpeg" else _png_bytes(image)
    page.insert_image(PAGE_RECT, stream=stream)


def linear_gradient_array(size: tuple[int, int], color_a: tuple[int, int, int], color_b: tuple[int, int, int], angle_deg: float = 0.0) -> np.ndarray:
    width, height = size
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, width), np.linspace(0.0, 1.0, height))
    angle = np.deg2rad(angle_deg)
    projection = xs * np.cos(angle) + ys * np.sin(angle)
    projection = (projection - projection.min()) / max(1e-6, projection.max() - projection.min())
    color_a_arr = np.array(color_a, dtype=np.float64)
    color_b_arr = np.array(color_b, dtype=np.float64)
    gradient = color_a_arr[None, None, :] + projection[:, :, None] * (color_b_arr - color_a_arr)[None, None, :]
    return np.clip(gradient, 0, 255).astype(np.uint8)


def radial_gradient_array(size: tuple[int, int], color_center: tuple[int, int, int], color_edge: tuple[int, int, int]) -> np.ndarray:
    width, height = size
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, width), np.linspace(-1.0, 1.0, height))
    distance = np.sqrt(xs**2 + ys**2)
    distance = np.clip(distance / distance.max(), 0.0, 1.0)
    color_center_arr = np.array(color_center, dtype=np.float64)
    color_edge_arr = np.array(color_edge, dtype=np.float64)
    gradient = color_center_arr[None, None, :] + distance[:, :, None] * (color_edge_arr - color_center_arr)[None, None, :]
    return np.clip(gradient, 0, 255).astype(np.uint8)


def synthetic_photo_texture(size: tuple[int, int], seed: int = 7) -> np.ndarray:
    """Multi-octave value noise standing in for a busy photo background.

    Real photographs are avoided on purpose (no external assets, fully
    reproducible); this produces the same kind of high local-texture region
    that trips up Telea-style diffusion inpainting.
    """

    rng = np.random.default_rng(seed)
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.float64)
    base_color = np.array([120.0, 95.0, 70.0])  # warm "wood/terrain" palette
    amplitude = 70.0
    scale = 10
    for _ in range(3):
        small = rng.uniform(-1.0, 1.0, size=(max(2, height // scale), max(2, width // scale), 3))
        layer = np.array(Image.fromarray(((small + 1) * 127.5).astype(np.uint8)).resize((width, height), Image.BICUBIC), dtype=np.float64)
        layer = (layer / 127.5) - 1.0
        canvas += layer * amplitude
        amplitude *= 0.45
        scale = max(1, scale // 2)
    canvas = base_color[None, None, :] + canvas
    return np.clip(canvas, 0, 255).astype(np.uint8)


def build_solid_background() -> None:
    """Solid colour background: opencv-fast prefill-path baseline."""

    doc, page = _new_doc()
    page.draw_rect(PAGE_RECT, color=None, fill=(0.90, 0.93, 0.97), fill_opacity=1)
    page.insert_textbox(
        fitz.Rect(70, 60, 890, 150),
        "Quarterly Planning Overview",
        fontsize=30,
        fontname="helv",
        color=(0.08, 0.12, 0.22),
    )
    page.insert_textbox(
        fitz.Rect(70, 190, 620, 340),
        LOREM,
        fontsize=16,
        fontname="helv",
        color=(0.15, 0.17, 0.20),
        align=0,
    )
    _save(doc, "solid_background.pdf")


def build_gradient_linear() -> None:
    """Linear gradient background: exercises smooth_gradient_* parameters."""

    doc, page = _new_doc()
    gradient = linear_gradient_array((320, 180), (245, 205, 120), (60, 90, 190), angle_deg=25.0)
    _insert_background_image(page, Image.fromarray(gradient, mode="RGB"))
    page.insert_textbox(
        fitz.Rect(80, 70, 860, 160),
        "Regional Revenue Trend",
        fontsize=28,
        fontname="helv",
        color=(1, 1, 1),
    )
    page.insert_textbox(
        fitz.Rect(80, 220, 560, 340),
        "Linear gradient backdrop used to validate the smooth background "
        "reconstruction path when text sits across a colour ramp.",
        fontsize=15,
        fontname="helv",
        color=(0.05, 0.05, 0.08),
    )
    _save(doc, "gradient_linear.pdf")


def build_gradient_radial() -> None:
    """Radial gradient background: same purpose as the linear case, different topology."""

    doc, page = _new_doc()
    gradient = radial_gradient_array((320, 180), (255, 250, 235), (40, 60, 95))
    _insert_background_image(page, Image.fromarray(gradient, mode="RGB"))
    page.insert_textbox(
        fitz.Rect(180, 90, 780, 180),
        "Spotlight Announcement",
        fontsize=30,
        fontname="helv",
        color=(0.10, 0.10, 0.12),
    )
    page.insert_textbox(
        fitz.Rect(230, 380, 730, 470),
        "Radial gradient centred on the page to probe vignette-style backgrounds.",
        fontsize=15,
        fontname="helv",
        color=(0.95, 0.95, 0.95),
    )
    _save(doc, "gradient_radial.pdf")


def build_geometric_chart() -> None:
    """Bar-chart-like geometry: exercises structural line-region protection."""

    doc, page = _new_doc()
    page.draw_rect(PAGE_RECT, color=None, fill=(1, 1, 1), fill_opacity=1)
    page.insert_textbox(
        fitz.Rect(60, 40, 700, 90),
        "Monthly Active Users",
        fontsize=24,
        fontname="helv",
        color=(0.1, 0.1, 0.1),
    )

    axis_origin = (110.0, 470.0)
    axis_top = (110.0, 110.0)
    axis_right = (900.0, 470.0)
    page.draw_line(fitz.Point(*axis_origin), fitz.Point(*axis_top), color=(0.2, 0.2, 0.2), width=2)
    page.draw_line(fitz.Point(*axis_origin), fitz.Point(*axis_right), color=(0.2, 0.2, 0.2), width=2)

    bars = [140, 260, 190, 320, 250, 300, 210]
    colors = [
        (0.20, 0.45, 0.80),
        (0.90, 0.55, 0.15),
        (0.30, 0.65, 0.35),
        (0.75, 0.25, 0.35),
        (0.55, 0.35, 0.75),
        (0.20, 0.65, 0.65),
        (0.85, 0.75, 0.20),
    ]
    bar_width = 90.0
    gap = 20.0
    x = axis_origin[0] + 25.0
    for height, color in zip(bars, colors, strict=True):
        top = axis_origin[1] - height
        rect = fitz.Rect(x, top, x + bar_width, axis_origin[1])
        page.draw_rect(rect, color=(0.1, 0.1, 0.1), fill=color, fill_opacity=1, width=1)
        page.insert_text(fitz.Point(x + 18, top - 10), str(height), fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1))
        x += bar_width + gap

    page.insert_textbox(
        fitz.Rect(60, 480, 900, 520),
        "Jan  Feb  Mar  Apr  May  Jun  Jul",
        fontsize=14,
        fontname="helv",
        color=(0.2, 0.2, 0.2),
    )
    _save(doc, "geometric_chart.pdf")


def build_photo_texture() -> None:
    """Synthesised high-texture background: the main Telea-vs-LaMa gap scenario."""

    doc, page = _new_doc()
    texture = synthetic_photo_texture((320, 180), seed=7)
    _insert_background_image(page, Image.fromarray(texture, mode="RGB"), image_format="jpeg")
    page.draw_rect(fitz.Rect(60, 340, 620, 460), color=None, fill=(0, 0, 0), fill_opacity=0.35)
    page.insert_textbox(
        fitz.Rect(80, 360, 700, 440),
        "Field Report: Site Survey",
        fontsize=26,
        fontname="helv",
        color=(1, 1, 1),
    )
    page.insert_textbox(
        fitz.Rect(80, 60, 620, 130),
        "Caption overlaid on a noisy synthetic texture (no external photo assets).",
        fontsize=15,
        fontname="helv",
        color=(1, 1, 1),
    )
    _save(doc, "photo_texture.pdf")


def build_dark_background_light_text() -> None:
    """Dark background + light text: stresses the white-box degradation case."""

    doc, page = _new_doc()
    page.draw_rect(PAGE_RECT, color=None, fill=(0.06, 0.07, 0.10), fill_opacity=1)
    page.draw_rect(fitz.Rect(0, 0, PAGE_WIDTH, 6), color=None, fill=(0.30, 0.55, 0.95), fill_opacity=1)
    page.insert_textbox(
        fitz.Rect(80, 90, 880, 170),
        "Night Mode Dashboard",
        fontsize=30,
        fontname="helv",
        color=(0.95, 0.97, 1.0),
    )
    page.insert_textbox(
        fitz.Rect(80, 220, 640, 360),
        LOREM,
        fontsize=16,
        fontname="helv",
        color=(0.85, 0.87, 0.92),
    )
    _save(doc, "dark_background_light_text.pdf")


def build_table() -> None:
    """Simple ruled table: Phase 3 layout-analysis baseline."""

    doc, page = _new_doc()
    page.draw_rect(PAGE_RECT, color=None, fill=(1, 1, 1), fill_opacity=1)
    page.insert_textbox(
        fitz.Rect(60, 40, 700, 85),
        "Budget Summary",
        fontsize=24,
        fontname="helv",
        color=(0.1, 0.1, 0.1),
    )

    left, top, right, bottom = 80.0, 120.0, 880.0, 460.0
    rows = 6
    cols = 4
    row_height = (bottom - top) / rows
    col_width = (right - left) / cols

    for row_index in range(rows + 1):
        y = top + row_index * row_height
        page.draw_line(fitz.Point(left, y), fitz.Point(right, y), color=(0.3, 0.3, 0.3), width=1)
    for col_index in range(cols + 1):
        x = left + col_index * col_width
        page.draw_line(fitz.Point(x, top), fitz.Point(x, bottom), color=(0.3, 0.3, 0.3), width=1)

    header_rect = fitz.Rect(left, top, right, top + row_height)
    page.draw_rect(header_rect, color=None, fill=(0.85, 0.88, 0.94), fill_opacity=1)

    headers = ["Department", "Q1", "Q2", "Q3"]
    for col_index, header in enumerate(headers):
        cell = fitz.Rect(
            left + col_index * col_width,
            top,
            left + (col_index + 1) * col_width,
            top + row_height,
        )
        page.insert_textbox(cell + (8, 8, -8, -8), header, fontsize=14, fontname="helv", color=(0.1, 0.1, 0.1))

    body_rows = [
        ["Engineering", "120", "134", "141"],
        ["Sales", "88", "95", "101"],
        ["Marketing", "54", "60", "58"],
        ["Support", "37", "39", "41"],
        ["Operations", "62", "65", "70"],
    ]
    for row_offset, row_values in enumerate(body_rows, start=1):
        for col_index, value in enumerate(row_values):
            cell = fitz.Rect(
                left + col_index * col_width,
                top + row_offset * row_height,
                left + (col_index + 1) * col_width,
                top + (row_offset + 1) * row_height,
            )
            page.insert_textbox(cell + (8, 8, -8, -8), value, fontsize=13, fontname="helv", color=(0.15, 0.15, 0.15))

    _save(doc, "table.pdf")


def build_large_mask() -> None:
    """Single paragraph covering > 12% of the page: hits the auto white-box fallback."""

    doc, page = _new_doc()
    gradient = linear_gradient_array((320, 180), (230, 235, 240), (170, 185, 205), angle_deg=90.0)
    _insert_background_image(page, Image.fromarray(gradient, mode="RGB"))

    paragraph_rect = fitz.Rect(60, 130, 900, 430)
    long_text = " ".join([LOREM] * 4)
    used = page.insert_textbox(
        paragraph_rect,
        long_text,
        fontsize=17,
        fontname="helv",
        color=(0.10, 0.12, 0.18),
        lineheight=1.35,
    )
    if used < 0:
        raise RuntimeError("large_mask fixture text overflowed its textbox; shorten the paragraph")

    mask_ratio = (paragraph_rect.width * paragraph_rect.height) / PAGE_AREA
    print(f"  large_mask textbox area ratio: {mask_ratio:.4f} (target > 0.12)")
    _save(doc, "large_mask.pdf")


def main() -> None:
    build_solid_background()
    build_gradient_linear()
    build_gradient_radial()
    build_geometric_chart()
    build_photo_texture()
    build_dark_background_light_text()
    build_table()
    build_large_mask()


if __name__ == "__main__":
    main()
