#!/usr/bin/env python3
"""Phase 0 / P0 slide-quality benchmark: convert fixtures and score fidelity + editability.

For every PDF under ``tests/fixtures/`` (or ``--fixtures-dir``), this script:

1. Runs ``convert_pdf`` with the requested conversion options (default path
   unchanged: ``opencv-fast``, no generative backend required).
2. Renders the PPTX headlessly via LibreOffice -> PDF -> PNG.
3. Compares each slide to the source PDF page (SSIM, pixel error, edge F1,
   optional OCR text F1) and scores editability from PPTX object counts.

Fidelity and editability are written separately so two runs/configs can be
diffed without conflating "looks like the original" with "is editable".

Usage:

    python scripts/bench_slide.py
    python scripts/bench_slide.py --ocr off --json-out output/bench_slide.json
    python scripts/bench_slide.py --inpaint-engine white-box --fixtures-dir tests/fixtures

Requires LibreOffice (``soffice``) on PATH. OCR is optional (``--ocr auto|on|off``).
Pin ``Noto Sans CJK TC`` (``fonts-noto-cjk``) for stable CJK metrics; see
``docs/slide-quality.md``.

Interface note: ``--inpaint-engine`` and ``--extra-option KEY=VALUE`` keep the
surface simple so a parallel generative-inpainting backend can be measured
later without changing this script's contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the OCR probe from the inpaint bench so availability behaviour matches.
import bench_inpaint  # noqa: E402

from pdf2ppt.core import ConversionOptions  # noqa: E402
from pdf2ppt.pipeline import convert_pdf  # noqa: E402
from pdf2ppt.render_compare import (  # noqa: E402
    DEFAULT_COMPARE_DPI,
    DeckQualityReport,
    RenderCompareError,
    cjk_font_status,
    evaluate_pptx_against_pdf,
    find_libreoffice_executable,
    summarize_slide_scores,
)

DEFAULT_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
DEFAULT_JSON_OUT = REPO_ROOT / "output" / "bench_slide.json"
DEFAULT_WORK_DIR = REPO_ROOT / "output" / "bench_slide_work"


@dataclass(slots=True)
class BenchSlideConfig:
    fixtures_dir: Path
    work_dir: Path
    json_out: Path
    dpi: int
    ocr_mode: str
    ocr_lang: str
    ocr_model_root: Path | None
    convert_ocr_model_root: Path | None
    inpaint_engine: str
    keep_artifacts: bool
    fixture_names: tuple[str, ...]


def _parse_extra_option(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--extra-option expects KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"empty key in --extra-option {raw!r}")
    return key, value


def _apply_extra_options(options: ConversionOptions, extras: list[tuple[str, str]]) -> None:
    """Set ConversionOptions attributes from KEY=VALUE pairs (string/bool/int/float)."""

    for key, raw_value in extras:
        if not hasattr(options, key):
            raise SystemExit(f"Unknown ConversionOptions field via --extra-option: {key}")
        current = getattr(options, key)
        if isinstance(current, bool):
            value: Any = raw_value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw_value)
        elif isinstance(current, float):
            value = float(raw_value)
        elif isinstance(current, Path) or current is None and key.endswith("_path"):
            value = Path(raw_value)
        else:
            value = raw_value
        setattr(options, key, value)


def convert_fixture(
    pdf_path: Path,
    out_dir: Path,
    *,
    inpaint_engine: str,
    convert_ocr_model_root: Path | None,
    extras: list[tuple[str, str]],
) -> tuple[Path, Path, float]:
    """Convert one fixture PDF; return (pptx_path, report_path, seconds)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    pptx_path = out_dir / f"{pdf_path.stem}.pptx"
    report_path = out_dir / f"{pdf_path.stem}.report.json"
    options = ConversionOptions(
        input_path=pdf_path,
        output_path=pptx_path,
        report_path=report_path,
        inpaint_engine=inpaint_engine,
        # English fixtures; hybrid pages still need a working PaddleOCR stack.
        lang="en",
        # Default ConversionOptions.ocr_model_root is Path("model"); an empty or
        # incomplete local tree breaks hybrid fixtures. None uses the official
        # PaddleX download cache (same behaviour as a fresh install).
        ocr_model_root=convert_ocr_model_root,
    )
    _apply_extra_options(options, extras)
    started = time.perf_counter()
    convert_pdf(options)
    elapsed = time.perf_counter() - started
    return pptx_path, report_path, elapsed


def run_benchmark(config: BenchSlideConfig, *, extras: list[tuple[str, str]]) -> dict[str, Any]:
    if find_libreoffice_executable() is None:
        raise SystemExit(
            "LibreOffice (`soffice` / `libreoffice`) not found on PATH. "
            "Install libreoffice-impress to run the slide-quality benchmark."
        )

    fixture_paths = sorted(config.fixtures_dir.glob("*.pdf"))
    if config.fixture_names:
        wanted = set(config.fixture_names)
        fixture_paths = [path for path in fixture_paths if path.stem in wanted]
    if not fixture_paths:
        raise SystemExit(f"No fixture PDFs found under {config.fixtures_dir}")

    if config.ocr_mode == "off":
        ocr_availability = bench_inpaint.OcrAvailability(engine=None, reason="OCR disabled via --ocr=off")
    else:
        ocr_availability = bench_inpaint.probe_ocr_engine(
            lang=config.ocr_lang,
            model_root=config.ocr_model_root,
        )
        if ocr_availability.engine is None and config.ocr_mode == "on":
            raise SystemExit(f"--ocr=on but OCR is unavailable: {ocr_availability.reason}")

    font_ok, font_note = cjk_font_status()
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dpi": config.dpi,
        "inpaint_engine": config.inpaint_engine,
        "renderer": "libreoffice->pdf->png",
        "ocr": {"available": ocr_availability.engine is not None, "reason": ocr_availability.reason},
        "cjk_font_available": font_ok,
        "cjk_font_note": font_note,
        "cases": [],
    }

    config.work_dir.mkdir(parents=True, exist_ok=True)
    all_page_scores = []

    for pdf_path in fixture_paths:
        case_dir = config.work_dir / pdf_path.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        convert_seconds: float | None = None
        case_error: str | None = None
        case_payload: dict[str, Any] = {
            "name": pdf_path.stem,
            "source_pdf": str(pdf_path),
        }
        try:
            pptx_path, _report_path, convert_seconds = convert_fixture(
                pdf_path,
                case_dir,
                inpaint_engine=config.inpaint_engine,
                convert_ocr_model_root=config.convert_ocr_model_root,
                extras=extras,
            )
            deck: DeckQualityReport = evaluate_pptx_against_pdf(
                pptx_path,
                pdf_path,
                dpi=float(config.dpi),
                ocr_engine=ocr_availability.engine,
                work_dir=case_dir / "render",
            )
            case_payload.update(deck.to_dict())
            case_payload["convert_seconds"] = round(convert_seconds, 4)
            all_page_scores.extend(deck.pages)
        except (RenderCompareError, OSError, RuntimeError) as error:
            case_error = f"{type(error).__name__}: {error}"
            case_payload["error"] = case_error
            case_payload["convert_seconds"] = (
                round(convert_seconds, 4) if convert_seconds is not None else None
            )
        report["cases"].append(case_payload)

    report["summary"] = summarize_slide_scores(all_page_scores)
    report["summary"]["fixture_count"] = len(report["cases"])
    report["summary"]["failed_fixture_count"] = sum(1 for case in report["cases"] if "error" in case)
    return report


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Fixtures: {len(report['cases'])}  "
        f"OCR: {report['ocr']['available']}  "
        f"CJK font: {report['cjk_font_available']}  "
        f"engine: {report['inpaint_engine']}"
    ]
    if not report["ocr"]["available"]:
        lines.append(f"  (OCR reason: {report['ocr']['reason']})")
    lines.append(f"  ({report['cjk_font_note']})")

    header = (
        f"{'fixture':28s} {'ssim':>7s} {'mae':>8s} {'edgeF1':>7s} "
        f"{'txtF1':>7s} {'edit':>7s} {'raster%':>8s} {'native':>6s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for case in report["cases"]:
        if "error" in case:
            lines.append(f"{case['name']:28s} ERROR {case['error']}")
            continue
        pages = case.get("pages") or []
        if not pages:
            lines.append(f"{case['name']:28s} (no pages)")
            continue
        # Fixtures are single-page today; still average for robustness.
        ssim = sum(page["fidelity"]["ssim"] for page in pages) / len(pages)
        mae = sum(page["fidelity"]["pixel_mae"] for page in pages) / len(pages)
        edge = sum(page["fidelity"]["edge_f1"] for page in pages) / len(pages)
        text_values = [page["fidelity"]["text_f1"] for page in pages if page["fidelity"]["text_f1"] is not None]
        text_str = f"{sum(text_values) / len(text_values):.3f}" if text_values else "n/a"
        edit = sum(page["editability"]["score"] for page in pages) / len(pages)
        raster = sum(page["editability"]["raster_background_area_ratio"] for page in pages) / len(pages)
        native = sum(page["editability"]["native_object_count"] for page in pages) / len(pages)
        lines.append(
            f"{case['name']:28s} {ssim:7.4f} {mae:8.2f} {edge:7.4f} "
            f"{text_str:>7s} {edit:7.4f} {raster * 100:7.2f}% {native:6.1f}"
        )

    summary = report.get("summary") or {}
    fidelity = summary.get("fidelity") or {}
    editability = summary.get("editability") or {}
    if fidelity:
        lines.append("")
        lines.append(
            "summary: "
            f"ssim={fidelity.get('ssim_mean')}  "
            f"mae={fidelity.get('pixel_mae_mean')}  "
            f"edge_f1={fidelity.get('edge_f1_mean')}  "
            f"text_f1={fidelity.get('text_f1_mean')}  "
            f"edit={editability.get('score_mean')}  "
            f"raster={editability.get('raster_background_area_ratio_mean')}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--fixture", action="append", default=[], help="Limit to fixture stem(s); repeatable.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--dpi", type=int, default=DEFAULT_COMPARE_DPI)
    parser.add_argument(
        "--ocr",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto: try once, degrade if unavailable (default). "
        "on: fail if OCR cannot start. off: skip text F1.",
    )
    parser.add_argument("--ocr-lang", default="en")
    parser.add_argument(
        "--ocr-model-root",
        type=Path,
        default=None,
        help="Model root for the evaluation-time text-F1 OCR probe (not conversion).",
    )
    parser.add_argument(
        "--convert-ocr-model-root",
        type=Path,
        default=None,
        help="Model root passed to ConversionOptions during convert_pdf. "
        "Default None uses the official PaddleX cache (avoids a broken local model/ tree).",
    )
    parser.add_argument(
        "--inpaint-engine",
        default="opencv-fast",
        help="Passed to ConversionOptions.inpaint_engine so alternate backends "
        "(including a future generative inpainting path) can be benched.",
    )
    parser.add_argument(
        "--extra-option",
        action="append",
        default=[],
        type=_parse_extra_option,
        metavar="KEY=VALUE",
        help="Set an additional ConversionOptions field (repeatable).",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep PPTX/PDF/PNG under --work-dir (always kept; flag reserved for clarity).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchSlideConfig(
        fixtures_dir=args.fixtures_dir,
        work_dir=args.work_dir,
        json_out=args.json_out,
        dpi=args.dpi,
        ocr_mode=args.ocr,
        ocr_lang=args.ocr_lang,
        ocr_model_root=args.ocr_model_root,
        convert_ocr_model_root=args.convert_ocr_model_root,
        inpaint_engine=args.inpaint_engine,
        keep_artifacts=bool(args.keep_artifacts),
        fixture_names=tuple(args.fixture),
    )
    report = run_benchmark(config, extras=list(args.extra_option))
    config.json_out.parent.mkdir(parents=True, exist_ok=True)
    config.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_summary(report))
    print(f"\nJSON written to {config.json_out}")
    print(f"Artifacts under {config.work_dir}")
    failed = report["summary"].get("failed_fixture_count", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
