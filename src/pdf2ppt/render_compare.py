"""Slide-level fidelity and editability evaluation (Phase 0 / P0).

Render a converted PPTX headlessly (LibreOffice -> PDF -> PNG), compare each
slide image to the corresponding source PDF page, and score editability from
the PPTX structure itself.

Fidelity and editability are reported separately on purpose: a full-page raster
background can score high on SSIM while remaining hard to edit.

System dependencies (optional; conversion itself does not require them):

- LibreOffice (``soffice`` / ``libreoffice``) for PPTX -> PDF
- A CJK-capable font, ideally ``Noto Sans CJK TC`` (Debian/Ubuntu:
  ``fonts-noto-cjk``), so headless rendering matches the typefaces pdf2ppt
  writes into OOXML. LibreOffice font metrics still differ from Microsoft
  PowerPoint -- treat these scores as a reproducible CI baseline, not as a
  PowerPoint WYSIWYG guarantee, especially for CJK.

Heavy optional pieces (OCR for text F1) reuse the existing ``OcrEngine`` and
degrade to ``null`` when unavailable, matching ``scripts/bench_inpaint.py``.
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

logger = logging.getLogger(__name__)

# Full-slide pictures at or above this area ratio count as "raster background".
RASTER_BACKGROUND_AREA_RATIO = 0.85
DEFAULT_COMPARE_DPI = 144
DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS = 180
# Padding around PDF text boxes when scoring inpaint-region MAE (matches bench reports).
DEFAULT_INPAINT_REGION_MASK_PADDING_PX = 8
DEFAULT_INPAINT_BOUNDARY_RING_PX = 4
CJK_FONT_FAMILY = "Noto Sans CJK TC"
# Common install locations for Noto Sans CJK (TC / SC / JP share a TTC on Debian).
CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
)


class RenderCompareError(RuntimeError):
    """Raised when headless rendering or comparison cannot proceed."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FidelityMetrics:
    """How closely a rendered slide matches the source page visually / textually."""

    ssim: float
    pixel_mae: float
    pixel_mse: float
    edge_f1: float
    text_f1: float | None = None
    text_precision: float | None = None
    text_recall: float | None = None
    text_f1_note: str | None = None
    # Inpaint-region metrics vs a text-free PDF background ground truth.
    inpaint_region_mae: float | None = None
    inpaint_boundary_mae: float | None = None
    inpaint_mask_pixel_count: int | None = None
    inpaint_region_note: str | None = None

    def to_dict(self) -> dict[str, float | str | int | None]:
        return asdict(self)


@dataclass(slots=True)
class EditabilityMetrics:
    """How much of the slide is represented as native, editable objects."""

    text_box_count: int
    picture_count: int
    auto_shape_count: int
    table_count: int
    line_count: int
    group_count: int
    chart_count: int
    other_shape_count: int
    native_object_count: int
    raster_background_area_ratio: float
    score: float
    # Non-empty native table cells counted as text units (P1).
    nonempty_table_cell_count: int = 0
    # Non-empty paragraphs across text frames (P2: grouped lists count per item).
    paragraph_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class SlideQualityScore:
    """Per-slide score with fidelity and editability kept separate."""

    page_number: int
    fidelity: FidelityMetrics
    editability: EditabilityMetrics
    render_seconds: float | None = None
    compare_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "fidelity": self.fidelity.to_dict(),
            "editability": self.editability.to_dict(),
            "render_seconds": self.render_seconds,
            "compare_seconds": self.compare_seconds,
        }


@dataclass(slots=True)
class DeckQualityReport:
    """Aggregate report for one PPTX vs its source PDF."""

    source_pdf: str
    output_pptx: str
    dpi: float
    renderer: str
    cjk_font_available: bool
    cjk_font_note: str
    pages: list[SlideQualityScore] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_pdf": self.source_pdf,
            "output_pptx": self.output_pptx,
            "dpi": self.dpi,
            "renderer": self.renderer,
            "cjk_font_available": self.cjk_font_available,
            "cjk_font_note": self.cjk_font_note,
            "pages": [page.to_dict() for page in self.pages],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Font / LibreOffice discovery
# ---------------------------------------------------------------------------


def find_cjk_font_path() -> Path | None:
    """Return the first installed Noto CJK font path, if any."""

    for candidate in CJK_FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def cjk_font_status() -> tuple[bool, str]:
    """Whether a pinned CJK font is available, plus a human-readable note."""

    path = find_cjk_font_path()
    if path is not None:
        return True, (
            f"Using {path} for LibreOffice CJK metrics. "
            "LibreOffice font metrics still differ from Microsoft PowerPoint; "
            "treat scores as a reproducible baseline, not a PowerPoint WYSIWYG guarantee."
        )
    return False, (
        f"Pinned CJK font '{CJK_FONT_FAMILY}' not found. Install fonts-noto-cjk "
        "(Debian/Ubuntu) so headless rendering matches pdf2ppt's OOXML typefaces. "
        "Scores remain valid for regression diffs on the same machine, but absolute "
        "values will drift across hosts without the font."
    )


def find_libreoffice_executable() -> Path | None:
    """Locate a LibreOffice headless binary, or ``None`` if unavailable."""

    for name in ("soffice", "libreoffice"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


def require_libreoffice() -> Path:
    executable = find_libreoffice_executable()
    if executable is None:
        raise RenderCompareError(
            "LibreOffice is required for PPTX rendering (install libreoffice-impress "
            "or ensure `soffice` is on PATH). Conversion itself does not need it."
        )
    return executable


# ---------------------------------------------------------------------------
# Image metric primitives (pure; unit-tested without LibreOffice)
# ---------------------------------------------------------------------------


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def resize_to_match(reference: Image.Image, candidate: Image.Image) -> Image.Image:
    """Resize ``candidate`` to ``reference`` size with Lanczos when they differ."""

    if candidate.size == reference.size:
        return candidate.convert("RGB")
    return candidate.convert("RGB").resize(reference.size, Image.Resampling.LANCZOS)


def compute_ssim(reference: Image.Image, candidate: Image.Image, *, data_range: float = 255.0) -> float:
    """Structural similarity on grayscale (Wang et al.), implemented without scikit-image.

    Returns a value in roughly ``[-1, 1]``; identical images yield ``1.0``.
    """

    matched = resize_to_match(reference, candidate)
    ref_gray = cv2.cvtColor(_to_rgb_array(reference), cv2.COLOR_RGB2GRAY).astype(np.float64)
    hyp_gray = cv2.cvtColor(_to_rgb_array(matched), cv2.COLOR_RGB2GRAY).astype(np.float64)

    # 11x11 Gaussian window, matching the common skimage defaults.
    kernel_size = 11
    sigma = 1.5
    window = cv2.getGaussianKernel(kernel_size, sigma)
    window = window @ window.T

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = cv2.filter2D(ref_gray, -1, window)
    mu2 = cv2.filter2D(hyp_gray, -1, window)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(ref_gray * ref_gray, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(hyp_gray * hyp_gray, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(ref_gray * hyp_gray, -1, window) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator
    return round(float(ssim_map.mean()), 6)


def compute_pixel_error(reference: Image.Image, candidate: Image.Image) -> dict[str, float]:
    """Mean absolute / squared error over RGB channels (0-255 scale)."""

    ref = _to_rgb_array(reference)
    hyp = _to_rgb_array(resize_to_match(reference, candidate))
    diff = ref.astype(np.float64) - hyp.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    return {
        "pixel_mae": round(mae, 4),
        "pixel_mse": round(mse, 4),
        "pixel_mae_normalized": round(mae / 255.0, 6),
    }


def compute_masked_pixel_mae(
    reference: Image.Image,
    candidate: Image.Image,
    mask_image: Image.Image,
) -> dict[str, float | int]:
    """Mean absolute RGB error inside a binary mask (0–255 scale)."""

    ref = _to_rgb_array(reference)
    hyp = _to_rgb_array(resize_to_match(reference, candidate))
    mask = np.array(mask_image.convert("L"), dtype=np.uint8)
    if mask.shape[:2] != ref.shape[:2]:
        mask = np.array(
            mask_image.convert("L").resize((ref.shape[1], ref.shape[0]), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    binary = mask > 0
    count = int(np.count_nonzero(binary))
    if count == 0:
        return {"masked_mae": 0.0, "mask_pixel_count": 0}
    diff = np.abs(ref.astype(np.float64) - hyp.astype(np.float64))
    mae = float(np.mean(diff[binary]))
    return {"masked_mae": round(mae, 4), "mask_pixel_count": count}


def compute_boundary_ring_mae(
    reference: Image.Image,
    candidate: Image.Image,
    mask_image: Image.Image,
    *,
    ring_px: int = DEFAULT_INPAINT_BOUNDARY_RING_PX,
) -> float:
    """MAE on a thin ring just inside the mask boundary (seam visibility proxy)."""

    mask = np.array(mask_image.convert("L"), dtype=np.uint8)
    ref = _to_rgb_array(reference)
    if mask.shape[:2] != ref.shape[:2]:
        mask = np.array(
            mask_image.convert("L").resize((ref.shape[1], ref.shape[0]), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary) or ring_px <= 0:
        return 0.0
    eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=max(1, ring_px))
    ring = (binary > 0) & (eroded == 0)
    if not np.any(ring):
        ring = binary > 0
    ring_image = Image.fromarray((ring.astype(np.uint8) * 255), mode="L")
    return float(compute_masked_pixel_mae(reference, candidate, ring_image)["masked_mae"])


def render_text_free_background(
    pdf_path: Path,
    page_index: int,
    *,
    dpi: float = DEFAULT_COMPARE_DPI,
) -> Image.Image:
    """Rasterize a page after deleting the text layer (images/graphics kept)."""

    with fitz.open(pdf_path) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise RenderCompareError(f"Page index {page_index} out of range for {pdf_path}")
        # Work on a copy so the caller's file is never mutated on disk.
        scratch = fitz.open()
        scratch.insert_pdf(document, from_page=page_index, to_page=page_index)
        page = scratch[0]
        # Transparent fill so we remove glyphs without painting white boxes over
        # gradients / photos (images + line art are preserved).
        page.add_redact_annot(page.rect, fill=False)  # type: ignore[arg-type]
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,  # type: ignore[attr-defined]
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,  # type: ignore[attr-defined]
        )
        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        scratch.close()
        return image


def build_pdf_text_mask(
    pdf_path: Path,
    page_index: int,
    *,
    image_size: tuple[int, int],
    padding_px: int = DEFAULT_INPAINT_REGION_MASK_PADDING_PX,
) -> Image.Image:
    """Binary mask covering native PDF text boxes, optionally padded in pixels."""

    from PIL import ImageDraw

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    with fitz.open(pdf_path) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise RenderCompareError(f"Page index {page_index} out of range for {pdf_path}")
        page = document[page_index]
        page_rect = page.rect
        scale_x = image_size[0] / max(1.0, float(page_rect.width))
        scale_y = image_size[1] / max(1.0, float(page_rect.height))
        for block in page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT).get("blocks", []):
            if block.get("type") != 0:
                continue
            x0, y0, x1, y1 = block["bbox"]
            box = (
                int(math.floor(x0 * scale_x)) - padding_px,
                int(math.floor(y0 * scale_y)) - padding_px,
                int(math.ceil(x1 * scale_x)) + padding_px,
                int(math.ceil(y1 * scale_y)) + padding_px,
            )
            clamped = (
                max(0, box[0]),
                max(0, box[1]),
                min(image_size[0], box[2]),
                min(image_size[1], box[3]),
            )
            if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
                draw.rectangle(clamped, fill=255)
    return mask


def extract_pptx_background_image(pptx_path: Path, *, slide_index: int = 0) -> Image.Image | None:
    """Return the largest picture on a slide (typically the full-page overlay background)."""

    presentation = Presentation(str(pptx_path))
    if slide_index < 0 or slide_index >= len(presentation.slides):
        return None
    slide = presentation.slides[slide_index]
    slide_width = int(presentation.slide_width or 1)
    slide_height = int(presentation.slide_height or 1)
    slide_area = float(max(1, slide_width) * max(1, slide_height))
    best_image: Image.Image | None = None
    best_ratio = 0.0
    for shape in slide.shapes:
        if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
            continue
        area_ratio = _shape_area_emu(shape) / slide_area
        if area_ratio < best_ratio:
            continue
        try:
            blob = shape.image.blob
        except Exception:  # pragma: no cover - defensive
            continue
        image = Image.open(io.BytesIO(blob)).convert("RGB")
        best_image = image
        best_ratio = area_ratio
    return best_image


def compute_inpaint_region_metrics(
    pdf_path: Path,
    pptx_path: Path,
    *,
    page_index: int = 0,
    dpi: float = DEFAULT_COMPARE_DPI,
    mask_padding_px: int = DEFAULT_INPAINT_REGION_MASK_PADDING_PX,
    boundary_ring_px: int = DEFAULT_INPAINT_BOUNDARY_RING_PX,
) -> dict[str, float | int | str | None]:
    """Compare PPTX background vs text-free PDF GT inside the text mask."""

    try:
        gt = render_text_free_background(pdf_path, page_index, dpi=dpi)
    except Exception as error:  # pragma: no cover - defensive
        return {
            "inpaint_region_mae": None,
            "inpaint_boundary_mae": None,
            "inpaint_mask_pixel_count": None,
            "inpaint_region_note": f"text-free GT failed: {type(error).__name__}: {error}",
        }
    background = extract_pptx_background_image(pptx_path, slide_index=page_index)
    if background is None:
        return {
            "inpaint_region_mae": None,
            "inpaint_boundary_mae": None,
            "inpaint_mask_pixel_count": None,
            "inpaint_region_note": "no full-page background picture in PPTX",
        }
    background = resize_to_match(gt, background)
    mask = build_pdf_text_mask(pdf_path, page_index, image_size=gt.size, padding_px=mask_padding_px)
    masked = compute_masked_pixel_mae(gt, background, mask)
    if int(masked["mask_pixel_count"]) == 0:
        return {
            "inpaint_region_mae": 0.0,
            "inpaint_boundary_mae": 0.0,
            "inpaint_mask_pixel_count": 0,
            "inpaint_region_note": "no native PDF text boxes for inpaint mask",
        }
    boundary = compute_boundary_ring_mae(gt, background, mask, ring_px=boundary_ring_px)
    return {
        "inpaint_region_mae": float(masked["masked_mae"]),
        "inpaint_boundary_mae": round(boundary, 4),
        "inpaint_mask_pixel_count": int(masked["mask_pixel_count"]),
        "inpaint_region_note": None,
    }


def compute_edge_similarity(
    reference: Image.Image,
    candidate: Image.Image,
    *,
    canny_low: int = 80,
    canny_high: int = 160,
) -> dict[str, float]:
    """Edge-map precision / recall / F1 using Canny on both images."""

    ref_gray = cv2.cvtColor(_to_rgb_array(reference), cv2.COLOR_RGB2GRAY)
    hyp_gray = cv2.cvtColor(_to_rgb_array(resize_to_match(reference, candidate)), cv2.COLOR_RGB2GRAY)
    ref_edges = cv2.Canny(ref_gray, canny_low, canny_high) > 0
    hyp_edges = cv2.Canny(hyp_gray, canny_low, canny_high) > 0

    true_positive = int(np.count_nonzero(ref_edges & hyp_edges))
    false_positive = int(np.count_nonzero(~ref_edges & hyp_edges))
    false_negative = int(np.count_nonzero(ref_edges & ~hyp_edges))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "edge_precision": round(precision, 6),
        "edge_recall": round(recall, 6),
        "edge_f1": round(f1, 6),
    }


_TOKEN_RE = re.compile(r"[\w\u3400-\u9fff\uf900-\ufaff]+", re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    """Lowercase alphanumeric / CJK tokens for bag-of-tokens F1."""

    return [token.lower() for token in _TOKEN_RE.findall(text)]


def text_token_f1(reference_text: str, candidate_text: str) -> dict[str, float]:
    """Bag-of-tokens precision / recall / F1 between two strings."""

    ref_tokens = tokenize_text(reference_text)
    hyp_tokens = tokenize_text(candidate_text)
    if not ref_tokens and not hyp_tokens:
        return {"text_precision": 1.0, "text_recall": 1.0, "text_f1": 1.0}
    if not ref_tokens or not hyp_tokens:
        return {"text_precision": 0.0, "text_recall": 0.0, "text_f1": 0.0}

    ref_counts: dict[str, int] = {}
    hyp_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in hyp_tokens:
        hyp_counts[token] = hyp_counts.get(token, 0) + 1

    overlap = sum(min(ref_counts[token], hyp_counts.get(token, 0)) for token in ref_counts)
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "text_precision": round(precision, 6),
        "text_recall": round(recall, 6),
        "text_f1": round(f1, 6),
    }


def extract_pdf_page_text(pdf_path: Path, page_index: int) -> str:
    """Ground-truth text from the source PDF page (native extraction)."""

    with fitz.open(pdf_path) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise RenderCompareError(f"Page index {page_index} out of range for {pdf_path}")
        return document[page_index].get_text("text")


def ocr_image_text(ocr_engine: Any, image: Image.Image, *, page_number: int = 1) -> str:
    """Concatenate OCR block texts from an image using the production engine."""

    page_data = ocr_engine.extract_text_blocks(image, page_number)
    return "\n".join(block.text for block in page_data.blocks)


# ---------------------------------------------------------------------------
# Editability from PPTX structure
# ---------------------------------------------------------------------------


def _shape_area_emu(shape: Any) -> float:
    width = float(getattr(shape, "width", 0) or 0)
    height = float(getattr(shape, "height", 0) or 0)
    return abs(width * height)


def _count_nonempty_table_cells(shape: Any) -> int:
    """Count cells with non-whitespace text in a native PowerPoint table shape."""

    if not getattr(shape, "has_table", False):
        return 0
    table = shape.table
    count = 0
    for row in table.rows:
        for cell in row.cells:
            if (cell.text or "").strip():
                count += 1
    return count


def _count_text_frame_paragraphs(shape: Any) -> int:
    """Count non-empty paragraphs (or bullet paras) inside a text-bearing shape."""

    if not getattr(shape, "has_text_frame", False) or not shape.has_text_frame:
        return 0
    count = 0
    for paragraph in shape.text_frame.paragraphs:
        text = (paragraph.text or "").strip()
        if text:
            count += 1
            continue
        # Empty run text can still be a real bullet placeholder paragraph.
        try:
            p_pr = paragraph._p.pPr  # noqa: SLF001
        except Exception:  # noqa: BLE001
            p_pr = None
        if p_pr is not None:
            from pptx.oxml.ns import qn

            if any(
                child.tag in {qn("a:buChar"), qn("a:buAutoNum"), qn("a:buBlip")} for child in p_pr
            ):
                count += 1
    return count


def inspect_slide_editability(slide: Any, *, slide_width: int, slide_height: int) -> EditabilityMetrics:
    """Count native objects and estimate raster-background coverage for one slide."""

    slide_area = float(max(1, slide_width) * max(1, slide_height))
    text_box_count = 0
    paragraph_count = 0
    picture_count = 0
    auto_shape_count = 0
    table_count = 0
    line_count = 0
    group_count = 0
    chart_count = 0
    other_shape_count = 0
    nonempty_table_cell_count = 0
    raster_background_area = 0.0

    for shape in slide.shapes:
        shape_type = shape.shape_type
        if shape_type == MSO_SHAPE_TYPE.TEXT_BOX:
            text_box_count += 1
            paragraph_count += _count_text_frame_paragraphs(shape)
        elif shape_type == MSO_SHAPE_TYPE.PICTURE:
            picture_count += 1
            area_ratio = _shape_area_emu(shape) / slide_area
            if area_ratio >= RASTER_BACKGROUND_AREA_RATIO:
                raster_background_area = max(raster_background_area, area_ratio)
        elif shape_type == MSO_SHAPE_TYPE.TABLE:
            table_count += 1
            nonempty_table_cell_count += _count_nonempty_table_cells(shape)
        elif shape_type == MSO_SHAPE_TYPE.GROUP:
            group_count += 1
        elif shape_type == MSO_SHAPE_TYPE.CHART:
            chart_count += 1
        elif shape_type in {MSO_SHAPE_TYPE.AUTO_SHAPE, MSO_SHAPE_TYPE.FREEFORM, MSO_SHAPE_TYPE.CALLOUT}:
            auto_shape_count += 1
        elif shape_type == MSO_SHAPE_TYPE.LINE:
            line_count += 1
        else:
            # Placeholders that carry text still count toward editability.
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                text = (shape.text_frame.text or "").strip()
                if text:
                    text_box_count += 1
                    paragraph_count += _count_text_frame_paragraphs(shape)
                else:
                    other_shape_count += 1
            else:
                other_shape_count += 1

    native_object_count = (
        text_box_count + auto_shape_count + table_count + line_count + group_count + chart_count
    )
    # Composite in [0, 1]. Text units = paragraphs (not raw text boxes) +
    # non-empty native table cells so grouped bullet lists are not penalized.
    # Structure still rewards shapes/tables; weights (0.40 / 0.45 / 0.15) match P1.
    text_units = max(paragraph_count, text_box_count) + nonempty_table_cell_count
    text_component = min(1.0, text_units / 6.0) * 0.40
    structure_units = (
        auto_shape_count + table_count * 4 + line_count * 0.5 + group_count + chart_count * 2
    )
    structure_component = min(1.0, structure_units / 8.0) * 0.45
    if text_units == 0 and native_object_count == 0:
        raster_penalty = raster_background_area * 0.40
    elif structure_units >= 2:
        raster_penalty = raster_background_area * 0.10
    else:
        raster_penalty = raster_background_area * 0.15
    score = max(0.0, min(1.0, text_component + structure_component + 0.15 - raster_penalty))

    return EditabilityMetrics(
        text_box_count=text_box_count,
        picture_count=picture_count,
        auto_shape_count=auto_shape_count,
        table_count=table_count,
        line_count=line_count,
        group_count=group_count,
        chart_count=chart_count,
        other_shape_count=other_shape_count,
        native_object_count=native_object_count,
        raster_background_area_ratio=round(min(1.0, raster_background_area), 6),
        score=round(score, 6),
        nonempty_table_cell_count=nonempty_table_cell_count,
        paragraph_count=paragraph_count,
    )


def inspect_pptx_editability(pptx_path: Path) -> list[EditabilityMetrics]:
    """Return per-slide editability metrics for a PPTX file."""

    presentation = Presentation(str(pptx_path))
    slide_width = int(presentation.slide_width or 1)
    slide_height = int(presentation.slide_height or 1)
    return [
        inspect_slide_editability(slide, slide_width=slide_width, slide_height=slide_height)
        for slide in presentation.slides
    ]


# ---------------------------------------------------------------------------
# Headless rendering: PPTX -> PDF -> PNG
# ---------------------------------------------------------------------------


def render_pdf_page_image(pdf_path: Path, page_index: int, *, dpi: float = DEFAULT_COMPARE_DPI) -> Image.Image:
    """Rasterize one PDF page with PyMuPDF (same path the pipeline uses)."""

    with fitz.open(pdf_path) as document:
        if page_index < 0 or page_index >= document.page_count:
            raise RenderCompareError(f"Page index {page_index} out of range for {pdf_path}")
        page = document[page_index]
        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Kill ``process`` and any children started in its process group/session."""

    if process.poll() is not None:
        return
    pid = process.pid
    try:
        if sys.platform == "win32":
            # /T terminates the whole tree rooted at pid.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False,
                capture_output=True,
                timeout=30,
            )
        else:
            # start_new_session=True makes the child the leader of its own group.
            os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError) as error:
        logger.debug("Process-tree kill for pid=%s failed: %s", pid, error)
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=5)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass


def _popen_kwargs_for_process_group() -> dict[str, Any]:
    """Start LibreOffice in its own process group so timeouts can reap orphans."""

    if sys.platform == "win32":
        # CREATE_NEW_PROCESS_GROUP is required for tree kill via taskkill /T.
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def run_subprocess_with_process_group(
    command: list[str],
    *,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` in a new process group; kill the whole tree on timeout/error."""

    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
        "cwd": str(cwd) if cwd is not None else None,
        **_popen_kwargs_for_process_group(),
    }
    process = subprocess.Popen(command, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _kill_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=timeout_seconds,
            output=stdout,
            stderr=stderr,
        ) from error
    except Exception:
        _kill_process_tree(process)
        raise
    return subprocess.CompletedProcess(
        args=command,
        returncode=int(process.returncode or 0),
        stdout=stdout or "",
        stderr=stderr or "",
    )


def convert_pptx_to_pdf(
    pptx_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: int = DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS,
    libreoffice: Path | None = None,
) -> Path:
    """Convert PPTX to PDF via LibreOffice headless. Returns the PDF path."""

    pptx_path = pptx_path.resolve()
    if not pptx_path.exists():
        raise RenderCompareError(f"PPTX does not exist: {pptx_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = libreoffice or require_libreoffice()

    # Isolated user profile avoids "profile locked" failures in CI / parallel runs.
    with tempfile.TemporaryDirectory(prefix="pdf2ppt-lo-profile-") as profile_dir:
        env = os.environ.copy()
        # Prefer the pinned family when present; LibreOffice still falls back
        # to its own substitutes if a face is missing.
        command = [
            str(executable),
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--norestore",
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(pptx_path),
        ]
        logger.info("LibreOffice convert: %s", " ".join(command))
        try:
            completed = run_subprocess_with_process_group(
                command,
                timeout_seconds=float(timeout_seconds),
                env=env,
            )
        except subprocess.TimeoutExpired as error:
            raise RenderCompareError(
                f"LibreOffice timed out after {timeout_seconds}s converting {pptx_path}"
            ) from error

    pdf_path = output_dir / f"{pptx_path.stem}.pdf"
    if completed.returncode != 0 or not pdf_path.exists():
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RenderCompareError(f"LibreOffice failed to convert {pptx_path}: {detail}")
    return pdf_path


def render_pptx_pages(
    pptx_path: Path,
    *,
    dpi: float = DEFAULT_COMPARE_DPI,
    work_dir: Path | None = None,
    timeout_seconds: int = DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS,
) -> list[Image.Image]:
    """Render every slide of a PPTX to RGB images via LibreOffice -> PDF -> PNG."""

    owns_work_dir = work_dir is None
    root = Path(tempfile.mkdtemp(prefix="pdf2ppt-render-")) if owns_work_dir else work_dir
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    try:
        pdf_path = convert_pptx_to_pdf(pptx_path, root, timeout_seconds=timeout_seconds)
        with fitz.open(pdf_path) as document:
            page_count = document.page_count
        return [render_pdf_page_image(pdf_path, index, dpi=dpi) for index in range(page_count)]
    finally:
        if owns_work_dir:
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# High-level comparison
# ---------------------------------------------------------------------------


def compare_page_images(
    reference: Image.Image,
    candidate: Image.Image,
    *,
    reference_text: str | None = None,
    candidate_text: str | None = None,
    text_f1_note: str | None = None,
    inpaint_region_mae: float | None = None,
    inpaint_boundary_mae: float | None = None,
    inpaint_mask_pixel_count: int | None = None,
    inpaint_region_note: str | None = None,
) -> FidelityMetrics:
    """Compute fidelity metrics for one reference/candidate image pair."""

    matched = resize_to_match(reference, candidate)
    ssim = compute_ssim(reference, matched)
    pixel = compute_pixel_error(reference, matched)
    edges = compute_edge_similarity(reference, matched)

    text_f1: float | None = None
    text_precision: float | None = None
    text_recall: float | None = None
    note = text_f1_note
    if reference_text is not None and candidate_text is not None:
        text_scores = text_token_f1(reference_text, candidate_text)
        text_f1 = text_scores["text_f1"]
        text_precision = text_scores["text_precision"]
        text_recall = text_scores["text_recall"]
    elif note is None:
        note = "text F1 skipped (no reference/candidate text provided)"

    return FidelityMetrics(
        ssim=ssim,
        pixel_mae=pixel["pixel_mae"],
        pixel_mse=pixel["pixel_mse"],
        edge_f1=edges["edge_f1"],
        text_f1=text_f1,
        text_precision=text_precision,
        text_recall=text_recall,
        text_f1_note=note,
        inpaint_region_mae=inpaint_region_mae,
        inpaint_boundary_mae=inpaint_boundary_mae,
        inpaint_mask_pixel_count=inpaint_mask_pixel_count,
        inpaint_region_note=inpaint_region_note,
    )


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 6)


def summarize_slide_scores(pages: list[SlideQualityScore]) -> dict[str, Any]:
    """Mean fidelity / editability across pages for JSON summaries and diffs."""

    if not pages:
        return {
            "page_count": 0,
            "fidelity": {},
            "editability": {},
        }
    return {
        "page_count": len(pages),
        "fidelity": {
            "ssim_mean": round(sum(page.fidelity.ssim for page in pages) / len(pages), 6),
            "pixel_mae_mean": round(sum(page.fidelity.pixel_mae for page in pages) / len(pages), 4),
            "pixel_mse_mean": round(sum(page.fidelity.pixel_mse for page in pages) / len(pages), 4),
            "edge_f1_mean": round(sum(page.fidelity.edge_f1 for page in pages) / len(pages), 6),
            "text_f1_mean": _mean_optional([page.fidelity.text_f1 for page in pages]),
            "inpaint_region_mae_mean": _mean_optional(
                [page.fidelity.inpaint_region_mae for page in pages]
            ),
            "inpaint_boundary_mae_mean": _mean_optional(
                [page.fidelity.inpaint_boundary_mae for page in pages]
            ),
        },
        "editability": {
            "score_mean": round(sum(page.editability.score for page in pages) / len(pages), 6),
            "native_object_count_mean": round(
                sum(page.editability.native_object_count for page in pages) / len(pages), 3
            ),
            "raster_background_area_ratio_mean": round(
                sum(page.editability.raster_background_area_ratio for page in pages) / len(pages), 6
            ),
            "text_box_count_total": sum(page.editability.text_box_count for page in pages),
            "paragraph_count_total": sum(page.editability.paragraph_count for page in pages),
            "nonempty_table_cell_count_total": sum(
                page.editability.nonempty_table_cell_count for page in pages
            ),
            "auto_shape_count_total": sum(page.editability.auto_shape_count for page in pages),
            "table_count_total": sum(page.editability.table_count for page in pages),
        },
    }


def evaluate_pptx_against_pdf(
    pptx_path: Path,
    pdf_path: Path,
    *,
    dpi: float = DEFAULT_COMPARE_DPI,
    ocr_engine: Any | None = None,
    work_dir: Path | None = None,
    timeout_seconds: int = DEFAULT_LIBREOFFICE_TIMEOUT_SECONDS,
) -> DeckQualityReport:
    """Render ``pptx_path`` and compare each slide to ``pdf_path`` pages.

    ``ocr_engine`` is optional. When provided, text F1 uses native PDF text as
    ground truth and OCR on the rendered slide as the hypothesis -- so future
    backends (vector nativization, generative inpainting, VLM planning) can be
    measured with the same call signature.
    """

    pptx_path = pptx_path.resolve()
    pdf_path = pdf_path.resolve()
    font_ok, font_note = cjk_font_status()
    editability_per_slide = inspect_pptx_editability(pptx_path)

    rendered_pages = render_pptx_pages(
        pptx_path,
        dpi=dpi,
        work_dir=work_dir,
        timeout_seconds=timeout_seconds,
    )

    with fitz.open(pdf_path) as document:
        pdf_page_count = document.page_count
    if len(rendered_pages) != pdf_page_count:
        logger.warning(
            "Slide count (%s) != PDF page count (%s); comparing the overlapping prefix",
            len(rendered_pages),
            pdf_page_count,
        )
    page_count = min(len(rendered_pages), pdf_page_count, len(editability_per_slide))

    pages: list[SlideQualityScore] = []
    for index in range(page_count):
        reference = render_pdf_page_image(pdf_path, index, dpi=dpi)
        candidate = rendered_pages[index]
        reference_text = extract_pdf_page_text(pdf_path, index)
        candidate_text: str | None = None
        text_note: str | None = None
        if ocr_engine is not None:
            try:
                candidate_text = ocr_image_text(ocr_engine, candidate, page_number=index + 1)
            except Exception as error:  # pragma: no cover - defensive
                text_note = f"OCR failed: {type(error).__name__}: {error}"
        else:
            text_note = "OCR unavailable; text F1 not computed"

        inpaint_metrics = compute_inpaint_region_metrics(
            pdf_path,
            pptx_path,
            page_index=index,
            dpi=dpi,
        )
        region_mae = inpaint_metrics["inpaint_region_mae"]
        boundary_mae = inpaint_metrics["inpaint_boundary_mae"]
        mask_count = inpaint_metrics["inpaint_mask_pixel_count"]
        region_note = inpaint_metrics["inpaint_region_note"]
        fidelity = compare_page_images(
            reference,
            candidate,
            reference_text=reference_text if candidate_text is not None else None,
            candidate_text=candidate_text,
            text_f1_note=text_note,
            inpaint_region_mae=float(region_mae) if isinstance(region_mae, (int, float)) else None,
            inpaint_boundary_mae=(
                float(boundary_mae) if isinstance(boundary_mae, (int, float)) else None
            ),
            inpaint_mask_pixel_count=int(mask_count) if isinstance(mask_count, int) else None,
            inpaint_region_note=str(region_note) if isinstance(region_note, str) else None,
        )
        pages.append(
            SlideQualityScore(
                page_number=index + 1,
                fidelity=fidelity,
                editability=editability_per_slide[index],
            )
        )

    report = DeckQualityReport(
        source_pdf=str(pdf_path),
        output_pptx=str(pptx_path),
        dpi=dpi,
        renderer="libreoffice->pdf->png",
        cjk_font_available=font_ok,
        cjk_font_note=font_note,
        pages=pages,
        summary=summarize_slide_scores(pages),
    )
    return report
