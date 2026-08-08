#!/usr/bin/env python3
"""Phase 0 quality baseline: benchmark background-inpainting engines.

For every fixture PDF under ``tests/fixtures/`` and every requested engine,
this script renders the page, builds the text-removal mask with the exact
same helper the production pipeline uses, runs the engine, and reports four
metrics:

1. ``residual_text_detection_rate`` -- re-run OCR detection over the
   inpainted page and measure how much of the original mask area is still
   covered by newly-detected text. 0 is ideal. ``null`` when OCR is
   unavailable in the current environment (see ``--ocr``).
2. ``boundary_discontinuity`` -- ratio of the Sobel gradient magnitude in a
   thin band straddling the mask contour to the gradient magnitude in the
   surrounding context ring. Values near 1 mean the seam blends into the
   local background texture; large values mean a visible seam.
3. ``outside_mask_pixel_diff`` -- pixel-exact comparison of everything
   *outside* the mask between the original render and the inpainted result.
   This is the regression guardrail future phases depend on.
4. ``seconds_per_page`` -- wall-clock time for the ``inpaint()`` call.

Usage:

    python scripts/bench_inpaint.py
    python scripts/bench_inpaint.py --engines white-box opencv-fast --ocr off
    python scripts/bench_inpaint.py --json-out out/bench.json --grid-dir out/grids

Only ``white-box`` and ``opencv-fast`` are meant to run in environments
without GPU/model weights (the default). Do not pass ``lama-*`` engines here
unless the LaMa runtime is actually installed -- they will raise.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pdf2ppt.block_analysis import map_blocks_to_page_coordinates, render_page_image  # noqa: E402
from pdf2ppt.core import ConversionOptions, OcrInitializationError, OcrProcessingError  # noqa: E402
from pdf2ppt.inpainting_engines import BackgroundInpaintingEngine  # noqa: E402
from pdf2ppt.inpainting_masks import build_text_mask_image, mask_area_ratio  # noqa: E402
from pdf2ppt.inpainting_overlay import resolve_background_inpainting_engine  # noqa: E402
from pdf2ppt.native_extraction import extract_native_text_blocks  # noqa: E402
from pdf2ppt.ocr import OcrEngine  # noqa: E402

DEFAULT_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
DEFAULT_JSON_OUT = REPO_ROOT / "output" / "bench_inpaint.json"
DEFAULT_GRID_DIR = REPO_ROOT / "output" / "bench_inpaint_grids"
DEFAULT_ENGINES = ("white-box", "opencv-fast")
DEFAULT_DPI = 144
DEFAULT_PADDING_PX = 6
# Engines with a byte-exact "leave everything outside the mask alone"
# contract. opencv-fast intentionally feathers a few pixels beyond the raw
# mask (gaussian blend seam smoothing, structural line restoration) so it is
# excluded from the strict guard by default -- see README note in the final
# report for details.
STRICT_OUTSIDE_MASK_ENGINES = ("white-box",)


# ---------------------------------------------------------------------------
# Metric primitives (pure functions, independently unit-tested).
# ---------------------------------------------------------------------------


def outside_mask_pixel_diff(
    original_image: Image.Image,
    inpainted_image: Image.Image,
    mask_array: np.ndarray,
) -> dict[str, float | int]:
    """Pixel-exact diff of everything outside ``mask_array``.

    Returns a small stats dict rather than a bool so the benchmark can report
    "how far off" an engine is even when it does not hit strict zero.
    """

    original = np.array(original_image.convert("RGB"), dtype=np.int16)
    inpainted = np.array(inpainted_image.convert("RGB"), dtype=np.int16)
    outside = mask_array == 0
    outside_pixel_count = int(np.count_nonzero(outside))
    if outside_pixel_count == 0:
        return {
            "max_abs_diff": 0,
            "changed_pixel_count": 0,
            "changed_pixel_ratio": 0.0,
            "outside_pixel_count": 0,
        }

    diff = np.abs(original - inpainted)[outside]
    changed = np.any(diff > 0, axis=-1)
    return {
        "max_abs_diff": int(diff.max()),
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_pixel_ratio": round(float(np.count_nonzero(changed)) / float(outside_pixel_count), 6),
        "outside_pixel_count": outside_pixel_count,
    }


def assert_outside_mask_unchanged(
    original_image: Image.Image,
    inpainted_image: Image.Image,
    mask_array: np.ndarray,
    *,
    tolerance: int = 0,
) -> dict[str, float | int]:
    """Regression guardrail: raise if pixels outside ``mask_array`` moved.

    Returns the same stats dict as :func:`outside_mask_pixel_diff` on
    success so callers can log it, and raises ``AssertionError`` with the
    offending stats otherwise.
    """

    stats = outside_mask_pixel_diff(original_image, inpainted_image, mask_array)
    if stats["max_abs_diff"] > tolerance:
        raise AssertionError(
            "Non-masked pixels changed: "
            f"max_abs_diff={stats['max_abs_diff']} "
            f"changed_pixel_count={stats['changed_pixel_count']} "
            f"changed_pixel_ratio={stats['changed_pixel_ratio']} "
            f"(tolerance={tolerance})"
        )
    return stats


def mask_boundary_discontinuity(
    inpainted_image: Image.Image,
    mask_array: np.ndarray,
    *,
    band_px: int = 3,
    context_px: int = 10,
) -> float:
    """Ratio of edge energy at the mask seam vs. the surrounding background.

    ~1.0 means the seam is no rougher than ordinary background texture;
    values well above 1 indicate a visible discontinuity ("patched but with
    a seam"). Values are unbounded above; there is no meaningful upper cap.
    """

    mask_bool = mask_array > 0
    if not np.any(mask_bool):
        return 0.0

    gray = cv2.cvtColor(np.array(inpainted_image.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float64)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.hypot(grad_x, grad_y)

    band_kernel = np.ones((band_px * 2 + 1, band_px * 2 + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask_bool.astype(np.uint8), band_kernel) > 0
    eroded = cv2.erode(mask_bool.astype(np.uint8), band_kernel) > 0
    boundary_band = dilated & ~eroded
    if not np.any(boundary_band):
        boundary_band = mask_bool

    context_kernel = np.ones((context_px * 2 + 1, context_px * 2 + 1), dtype=np.uint8)
    context_ring = (cv2.dilate(mask_bool.astype(np.uint8), context_kernel) > 0) & ~dilated
    if not np.any(context_ring):
        context_ring = ~mask_bool

    boundary_energy = float(gradient_magnitude[boundary_band].mean())
    context_energy = float(gradient_magnitude[context_ring].mean())
    # On near-flat backgrounds (solid fills) the context gradient is ~0, which
    # would blow the ratio up to an uninformative huge number for what is
    # visually a near-invisible seam. Floor the denominator at a small
    # absolute gradient magnitude (on an 8-bit Sobel scale) so the ratio
    # stays meaningful instead of dividing by noise.
    denominator = max(context_energy, 1.0)
    return round(boundary_energy / denominator, 4)


def residual_text_detection_rate(
    ocr_engine: OcrEngine,
    inpainted_image: Image.Image,
    page_rect: fitz.Rect,
    mask_array: np.ndarray,
    *,
    page_number: int = 1,
) -> float:
    """Fraction of the original mask area still covered by OCR detections.

    Mirrors the coordinate round-trip the production pipeline uses
    (``OcrEngine.extract_text_blocks`` -> ``map_blocks_to_page_coordinates``
    -> ``build_text_mask_image``) so the comparison is apples-to-apples with
    how masks are built everywhere else in pdf2ppt.
    """

    original_mask_bool = mask_array > 0
    denominator = int(np.count_nonzero(original_mask_bool))
    if denominator == 0:
        return 0.0

    ocr_page_data = ocr_engine.extract_text_blocks(inpainted_image, page_number)
    if not ocr_page_data.blocks:
        return 0.0

    detected_blocks = map_blocks_to_page_coordinates(ocr_page_data.blocks, ocr_page_data.image.size, page_rect)
    detected_mask_image = build_text_mask_image(detected_blocks, inpainted_image.size, page_rect, padding_px=0)
    detected_mask_bool = np.array(detected_mask_image, dtype=np.uint8) > 0
    overlap = int(np.count_nonzero(detected_mask_bool & original_mask_bool))
    return round(overlap / denominator, 4)


# ---------------------------------------------------------------------------
# OCR availability probing (graceful degradation).
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OcrAvailability:
    engine: OcrEngine | None
    reason: str | None


class _OcrProbeTimeout(TimeoutError):
    pass


def probe_ocr_engine(
    *,
    lang: str = "en",
    model_root: Path | None = None,
    timeout_seconds: int = 60,
) -> OcrAvailability:
    """Attempt to construct+warm-up a PaddleOCR engine exactly once.

    PaddleOCR downloads model weights on first use; in a network-restricted
    sandbox that download reliably fails after ~30-40s. We bound the wait
    with SIGALRM (POSIX-only, main-thread only) so a pathological hang can't
    stall the whole benchmark, and turn any failure into a structured
    "unavailable" result instead of letting the script crash.
    """

    engine = OcrEngine(
        lang=lang,
        model_root=model_root,
        use_doc_orientation=False,
        use_textline_orientation=False,
        use_doc_unwarping=False,
        det_thresh=None,
        det_box_thresh=None,
        drop_score=None,
    )
    probe_image = Image.new("RGB", (64, 64), color=(255, 255, 255))

    supports_alarm = hasattr(signal, "SIGALRM")
    previous_handler = None
    if supports_alarm:

        def _on_timeout(signum: int, frame: Any) -> None:
            raise _OcrProbeTimeout(f"OCR engine warm-up exceeded {timeout_seconds}s")

        previous_handler = signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(timeout_seconds)

    try:
        engine.extract_text_blocks(probe_image, 1)
    except (OcrInitializationError, OcrProcessingError, _OcrProbeTimeout) as error:
        return OcrAvailability(engine=None, reason=str(error))
    except Exception as error:  # pragma: no cover - defensive catch-all
        return OcrAvailability(engine=None, reason=f"{type(error).__name__}: {error}")
    else:
        return OcrAvailability(engine=engine, reason=None)
    finally:
        if supports_alarm:
            signal.alarm(0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)


# ---------------------------------------------------------------------------
# Fixture loading + engine construction.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FixtureCase:
    name: str
    pdf_path: Path
    page_image: Image.Image
    mask_image: Image.Image
    mask_array: np.ndarray
    page_rect: fitz.Rect
    mask_area_ratio: float
    text_block_count: int


def load_fixture_case(pdf_path: Path, *, dpi: int, padding_px: int) -> FixtureCase:
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        text_blocks, _image_boxes = extract_native_text_blocks(page)
        page_image = render_page_image(page, dpi=dpi)
        mask_image = build_text_mask_image(text_blocks, page_image.size, page.rect, padding_px=padding_px)
        mask_array = np.array(mask_image, dtype=np.uint8)
        return FixtureCase(
            name=pdf_path.stem,
            pdf_path=pdf_path,
            page_image=page_image,
            mask_image=mask_image,
            mask_array=mask_array,
            page_rect=page.rect,
            mask_area_ratio=mask_area_ratio(mask_array),
            text_block_count=len(text_blocks),
        )
    finally:
        doc.close()


def build_engine(engine_name: str) -> BackgroundInpaintingEngine:
    """Resolve an engine the same way the pipeline does, via explicit request.

    Reuses ``resolve_background_inpainting_engine`` (inpainting_overlay.py)
    rather than instantiating engine classes directly, so the benchmark
    exercises the real selection path and its logging note.
    """

    options = ConversionOptions(
        input_path=Path("unused.pdf"),
        output_path=Path("unused.pptx"),
        report_path=Path("unused.report.json"),
        inpaint_engine=engine_name,
    )
    dummy_image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    dummy_mask = Image.new("L", (8, 8), color=0)
    engine, _note = resolve_background_inpainting_engine(dummy_image, dummy_mask, options)
    return engine


# ---------------------------------------------------------------------------
# Side-by-side comparison grids.
# ---------------------------------------------------------------------------


def _label(image: Image.Image, text: str) -> Image.Image:
    labeled = image.convert("RGB").copy()
    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - defensive
        font = None
    draw.rectangle((0, 0, labeled.width, 22), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return labeled


def save_comparison_grid(panels: list[tuple[str, Image.Image]], out_path: Path, *, thumb_width: int = 340) -> None:
    thumbs = []
    for title, image in panels:
        ratio = thumb_width / image.width
        thumb = image.convert("RGB").resize((thumb_width, max(1, int(image.height * ratio))), Image.LANCZOS)
        thumbs.append(_label(thumb, title))

    height = max(thumb.height for thumb in thumbs)
    total_width = sum(thumb.width for thumb in thumbs) + 8 * (len(thumbs) - 1)
    grid = Image.new("RGB", (total_width, height), color=(30, 30, 30))
    x = 0
    for thumb in thumbs:
        grid.paste(thumb, (x, 0))
        x += thumb.width + 8

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, format="PNG", optimize=True)


def mask_preview_image(page_image: Image.Image, mask_array: np.ndarray) -> Image.Image:
    overlay = np.array(page_image.convert("RGB"), dtype=np.uint8).copy()
    highlight = mask_array > 0
    overlay[highlight] = (0.35 * overlay[highlight] + 0.65 * np.array([255, 60, 60])).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB")


# ---------------------------------------------------------------------------
# Benchmark orchestration.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkConfig:
    fixtures_dir: Path
    engines: tuple[str, ...]
    dpi: int
    padding_px: int
    ocr_mode: str
    ocr_lang: str
    ocr_model_root: Path | None
    strict_engines: tuple[str, ...]
    json_out: Path
    grid_dir: Path | None


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    fixture_paths = sorted(config.fixtures_dir.glob("*.pdf"))
    if not fixture_paths:
        raise SystemExit(f"No fixture PDFs found under {config.fixtures_dir}")

    if config.ocr_mode == "off":
        ocr_availability = OcrAvailability(engine=None, reason="OCR disabled via --ocr=off")
    else:
        ocr_availability = probe_ocr_engine(lang=config.ocr_lang, model_root=config.ocr_model_root)
        if ocr_availability.engine is None and config.ocr_mode == "on":
            raise SystemExit(f"--ocr=on but OCR is unavailable: {ocr_availability.reason}")

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dpi": config.dpi,
        "padding_px": config.padding_px,
        "engines": list(config.engines),
        "ocr": {"available": ocr_availability.engine is not None, "reason": ocr_availability.reason},
        "cases": [],
    }

    for pdf_path in fixture_paths:
        case = load_fixture_case(pdf_path, dpi=config.dpi, padding_px=config.padding_px)
        case_report: dict[str, Any] = {
            "name": case.name,
            "mask_area_ratio": case.mask_area_ratio,
            "text_block_count": case.text_block_count,
            "engines": {},
        }
        panels: list[tuple[str, Image.Image]] = [
            ("original", case.page_image),
            ("mask", mask_preview_image(case.page_image, case.mask_array)),
        ]

        for engine_name in config.engines:
            engine = build_engine(engine_name)
            started_at = time.perf_counter()
            inpainted = engine.inpaint(case.page_image, case.mask_image)
            elapsed_seconds = time.perf_counter() - started_at

            outside_diff = outside_mask_pixel_diff(case.page_image, inpainted, case.mask_array)
            if engine_name in config.strict_engines:
                assert_outside_mask_unchanged(case.page_image, inpainted, case.mask_array, tolerance=0)

            boundary_score = mask_boundary_discontinuity(inpainted, case.mask_array)

            residual_rate: float | None = None
            residual_error: str | None = None
            if ocr_availability.engine is not None:
                try:
                    residual_rate = residual_text_detection_rate(
                        ocr_availability.engine,
                        inpainted,
                        case.page_rect,
                        case.mask_array,
                    )
                except Exception as error:  # pragma: no cover - defensive
                    residual_error = f"{type(error).__name__}: {error}"

            case_report["engines"][engine_name] = {
                "seconds_per_page": round(elapsed_seconds, 4),
                "boundary_discontinuity": boundary_score,
                "outside_mask_pixel_diff": outside_diff,
                "residual_text_detection_rate": residual_rate,
                "residual_text_detection_error": residual_error,
            }
            panels.append((engine_name, inpainted))

        report["cases"].append(case_report)

        if config.grid_dir is not None:
            save_comparison_grid(panels, config.grid_dir / f"{case.name}.png")

    return report


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Fixtures: {len(report['cases'])}  Engines: {', '.join(report['engines'])}  "
        f"OCR available: {report['ocr']['available']}"
    ]
    if not report["ocr"]["available"]:
        lines.append(f"  (OCR reason: {report['ocr']['reason']})")

    header = f"{'fixture':28s} {'mask%':>7s}"
    for engine_name in report["engines"]:
        header += f" | {engine_name:>28s}"
    lines.append(header)
    lines.append("-" * len(header))

    for case in report["cases"]:
        row = f"{case['name']:28s} {case['mask_area_ratio'] * 100:6.2f}%"
        for engine_name in report["engines"]:
            metrics = case["engines"][engine_name]
            residual = metrics["residual_text_detection_rate"]
            residual_str = "n/a" if residual is None else f"{residual:.3f}"
            row += (
                f" | b={metrics['boundary_discontinuity']:6.2f} "
                f"t={metrics['seconds_per_page']:5.2f}s "
                f"res={residual_str:>5s} "
                f"outΔ={metrics['outside_mask_pixel_diff']['max_abs_diff']:>3d}"
            )
        lines.append(row)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=list(DEFAULT_ENGINES),
        help="Engine names understood by ConversionOptions.inpaint_engine. "
        "Do not pass lama-* here unless model weights are actually installed.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--padding-px", type=int, default=DEFAULT_PADDING_PX)
    parser.add_argument(
        "--ocr",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto: try once, degrade gracefully if unavailable (default). "
        "on: fail the run if OCR cannot be initialized. off: skip the OCR metric entirely.",
    )
    parser.add_argument("--ocr-lang", default="en")
    parser.add_argument("--ocr-model-root", type=Path, default=None)
    parser.add_argument(
        "--strict-engines",
        nargs="*",
        default=list(STRICT_OUTSIDE_MASK_ENGINES),
        help="Engines for which the outside-mask-pixel-diff assertion is enforced with tolerance=0. "
        "opencv-fast is excluded by default because it intentionally feathers a few pixels beyond "
        "the raw mask boundary (blend seam smoothing, structural line restoration).",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--no-grid", action="store_true", help="Skip writing side-by-side comparison PNGs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        fixtures_dir=args.fixtures_dir,
        engines=tuple(args.engines),
        dpi=args.dpi,
        padding_px=args.padding_px,
        ocr_mode=args.ocr,
        ocr_lang=args.ocr_lang,
        ocr_model_root=args.ocr_model_root,
        strict_engines=tuple(args.strict_engines),
        json_out=args.json_out,
        grid_dir=None if args.no_grid else args.grid_dir,
    )

    report = run_benchmark(config)

    config.json_out.parent.mkdir(parents=True, exist_ok=True)
    config.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_summary(report))
    print(f"\nJSON written to {config.json_out}")
    if config.grid_dir is not None:
        print(f"Comparison grids written to {config.grid_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
