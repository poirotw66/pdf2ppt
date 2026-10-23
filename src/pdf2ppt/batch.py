"""Batch folder conversion: discover PDFs and orchestrate the existing pipeline.

Single-file conversion behaviour is unchanged; this module only loops over
discovered inputs, isolates failures, and writes a quality summary.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

from .core import ConversionOptions, InputValidationError
from .models import ConversionReport, PageResult
from .pipeline import convert_pdf

logger = logging.getLogger(__name__)

BatchStatus = Literal["ok", "failed", "skipped"]

# Exit codes for ``pdf2ppt batch`` (documented in docs/cli-reference.md).
EXIT_SUCCESS = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_USAGE = 2

SUMMARY_JSON_NAME = "summary.json"
SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_MD_NAME = "summary.md"

ConvertFn = Callable[..., ConversionReport]


@dataclass(slots=True)
class BatchFileResult:
    """Per-PDF outcome recorded in the batch summary."""

    file: str
    output: str | None
    pages: int | None
    status: BatchStatus
    seconds: float
    ocr_used: bool
    ocr_pages: int
    raster_fallback_pages: int
    warnings_count: int
    planner_cost_usd: float | None
    quality: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    ssim_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchSummary:
    """Aggregate batch run results."""

    input_dir: str
    output_dir: str
    results: list[BatchFileResult]
    totals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "results": [item.to_dict() for item in self.results],
            "totals": dict(self.totals),
        }


@dataclass(slots=True)
class DiscoveredPdf:
    """One PDF to convert, with a planned output PPTX path."""

    input_path: Path
    relative_key: str
    output_path: Path


def discover_pdfs(
    input_dir: Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Return PDF paths under ``input_dir`` in stable sorted order.

    Non-recursive by default. Matching is case-insensitive on the ``.pdf``
    suffix. Hidden files (name starts with ``.``) are skipped. When recursive,
    any path segment that starts with ``.`` is skipped.
    """
    if not input_dir.exists():
        raise InputValidationError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise InputValidationError(f"Input path is not a directory: {input_dir}")

    found: list[Path] = []
    if recursive:
        for path in input_dir.rglob("*"):
            if not path.is_file():
                continue
            if not _is_pdf_filename(path.name):
                continue
            try:
                relative = path.relative_to(input_dir)
            except ValueError:
                continue
            if any(part.startswith(".") for part in relative.parts):
                continue
            found.append(path)
    else:
        for path in input_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if _is_pdf_filename(path.name):
                found.append(path)

    return sorted(found, key=lambda item: str(item.relative_to(input_dir)).casefold())


def _is_pdf_filename(name: str) -> bool:
    return name.casefold().endswith(".pdf")


def plan_batch_outputs(
    pdfs: Sequence[Path],
    *,
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
) -> list[DiscoveredPdf]:
    """Map each PDF to an output PPTX path with deterministic collision handling.

    Recursive mode preserves relative subfolders under ``output_dir``. When two
    inputs would write the same PPTX (e.g. ``a.pdf`` and ``a.PDF``), later files
    get a ``__N`` stem suffix (``a__2.pptx``, ``a__3.pptx``, …).
    """
    used: set[Path] = set()
    planned: list[DiscoveredPdf] = []
    for pdf in pdfs:
        relative = pdf.relative_to(input_dir)
        relative_key = relative.as_posix()
        if recursive:
            relative_pptx = relative.with_suffix(".pptx")
        else:
            relative_pptx = Path(pdf.stem + ".pptx")
        desired = output_dir / relative_pptx
        output_path = _allocate_unique_path(desired, used)
        planned.append(
            DiscoveredPdf(
                input_path=pdf,
                relative_key=relative_key,
                output_path=output_path,
            )
        )
    return planned


def _allocate_unique_path(desired: Path, used: set[Path]) -> Path:
    if desired not in used:
        used.add(desired)
        return desired
    stem = desired.stem
    suffix = desired.suffix
    parent = desired.parent
    index = 2
    while True:
        candidate = parent / f"{stem}__{index}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def load_resume_ok_files(summary_path: Path) -> set[str]:
    """Return relative file keys marked ``ok`` in an existing summary.json."""
    if not summary_path.is_file():
        return set()
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Could not read resume summary %s: %s", summary_path, error)
        return set()
    results = payload.get("results")
    if not isinstance(results, list):
        return set()
    ok_files: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "ok" and isinstance(item.get("file"), str):
            ok_files.add(item["file"])
    return ok_files


def metrics_from_report(report: ConversionReport) -> dict[str, Any]:
    """Cheap quality / OCR / raster metrics derived from a ConversionReport."""
    pages = report.pages
    page_count = len(pages)
    ocr_pages = sum(1 for page in pages if _page_used_ocr(page))
    raster_pages = sum(1 for page in pages if page.background_mode == "full-page")
    warnings_count = sum(_page_warning_count(page) for page in pages)
    planner_cost: float | None = None
    if isinstance(report.layout_planner, dict):
        raw_cost = report.layout_planner.get("estimated_cost_usd")
        if isinstance(raw_cost, (int, float)):
            planner_cost = float(raw_cost)

    editable_ratios = [page.quality_score.editable_ratio for page in pages]
    text_confidences = [page.quality_score.text_confidence for page in pages]
    mean_editable = sum(editable_ratios) / page_count if page_count else 0.0
    mean_text_confidence = sum(text_confidences) / page_count if page_count else 0.0
    raster_fraction = raster_pages / page_count if page_count else 0.0
    # Cheap indicator in [0, 1]: prefer editable pages with confident text and few rasters.
    quality_score = max(
        0.0,
        min(1.0, 0.5 * mean_editable + 0.3 * mean_text_confidence + 0.2 * (1.0 - raster_fraction)),
    )
    return {
        "pages": page_count,
        "ocr_used": ocr_pages > 0,
        "ocr_pages": ocr_pages,
        "raster_fallback_pages": raster_pages,
        "warnings_count": warnings_count,
        "planner_cost_usd": planner_cost,
        "quality": {
            "score": round(quality_score, 4),
            "raster_fallback_fraction": round(raster_fraction, 4),
            "mean_editable_ratio": round(mean_editable, 4),
            "mean_text_confidence": round(mean_text_confidence, 4),
        },
    }


def _page_used_ocr(page: PageResult) -> bool:
    if page.page_kind in {"scanned", "hybrid"}:
        return True
    return any(block.source == "ocr" for block in page.text_blocks)


def _page_warning_count(page: PageResult) -> int:
    count = 0
    if page.fallback_reason:
        count += 1
    if page.vector_fallback_reason:
        count += 1
    if page.layout_fallback_reason:
        count += 1
    if page.background_inpaint_note:
        count += 1
    return count


def maybe_measure_ssim(
    *,
    pdf_path: Path,
    pptx_path: Path,
    measure_quality: bool,
) -> float | None:
    """Optionally compute mean page SSIM via LibreOffice; skip gracefully if missing."""
    if not measure_quality:
        return None
    try:
        from .render_compare import (
            evaluate_pptx_against_pdf,
            find_libreoffice_executable,
            summarize_slide_scores,
        )
    except ImportError:
        logger.warning("--measure-quality skipped: render_compare unavailable")
        return None
    if find_libreoffice_executable() is None:
        logger.warning("--measure-quality skipped: LibreOffice (soffice) not found")
        return None
    try:
        deck = evaluate_pptx_against_pdf(pptx_path, pdf_path)
        summary = summarize_slide_scores(deck.pages) if deck.pages else deck.summary
        fidelity = summary.get("fidelity") if isinstance(summary, dict) else None
        raw = fidelity.get("ssim_mean") if isinstance(fidelity, dict) else None
        if isinstance(raw, (int, float)):
            return float(raw)
        scores = [page.fidelity.ssim for page in deck.pages]
        if not scores:
            return None
        return sum(scores) / len(scores)
    except Exception as error:  # noqa: BLE001 — optional metric must never fail the batch
        logger.warning("--measure-quality failed for %s: %s", pptx_path, error)
        return None


def convert_one_planned(
    item: DiscoveredPdf,
    *,
    options_factory: Callable[[Path, Path, Path], ConversionOptions],
    convert_fn: ConvertFn = convert_pdf,
    measure_quality: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BatchFileResult:
    """Run the existing pipeline for one PDF and map the report into a batch row."""
    started = time.perf_counter()
    report_path = item.output_path.with_suffix(".report.json")
    try:
        options = options_factory(item.input_path, item.output_path, report_path)
        report = convert_fn(options, progress_callback=progress_callback)
        metrics = metrics_from_report(report)
        ssim_mean = maybe_measure_ssim(
            pdf_path=item.input_path,
            pptx_path=item.output_path,
            measure_quality=measure_quality,
        )
        return BatchFileResult(
            file=item.relative_key,
            output=_relative_or_str(item.output_path),
            pages=int(metrics["pages"]),
            status="ok",
            seconds=round(time.perf_counter() - started, 3),
            ocr_used=bool(metrics["ocr_used"]),
            ocr_pages=int(metrics["ocr_pages"]),
            raster_fallback_pages=int(metrics["raster_fallback_pages"]),
            warnings_count=int(metrics["warnings_count"]),
            planner_cost_usd=metrics["planner_cost_usd"],
            quality=dict(metrics["quality"]),
            ssim_mean=None if ssim_mean is None else round(ssim_mean, 4),
        )
    except Exception as error:  # noqa: BLE001 — isolate per-file failures
        return BatchFileResult(
            file=item.relative_key,
            output=_relative_or_str(item.output_path),
            pages=None,
            status="failed",
            seconds=round(time.perf_counter() - started, 3),
            ocr_used=False,
            ocr_pages=0,
            raster_fallback_pages=0,
            warnings_count=0,
            planner_cost_usd=None,
            quality={},
            error=str(error),
        )


def _relative_or_str(path: Path) -> str:
    return path.as_posix()


def _worker_convert(payload: dict[str, Any]) -> BatchFileResult:
    """Process-pool entry point (must be top-level for pickling)."""
    item = DiscoveredPdf(
        input_path=Path(payload["input_path"]),
        relative_key=payload["relative_key"],
        output_path=Path(payload["output_path"]),
    )
    options_kwargs = payload["options_kwargs"]

    def options_factory(input_path: Path, output_path: Path, report_path: Path) -> ConversionOptions:
        return ConversionOptions(
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            **options_kwargs,
        )

    return convert_one_planned(
        item,
        options_factory=options_factory,
        measure_quality=bool(payload.get("measure_quality", False)),
    )


def conversion_options_kwargs(options: ConversionOptions) -> dict[str, Any]:
    """Serializable ConversionOptions fields excluding per-file paths and runtime hooks."""
    skip = {
        "input_path",
        "output_path",
        "report_path",
        "layout_planner_budget",
        "layout_planner_transport",
        "generative_budget",
        "generative_residual_checker",
        "approved_ocr_blocks_by_page",
        "approved_ocr_image_size_by_page",
    }
    payload: dict[str, Any] = {}
    for name in ConversionOptions.__dataclass_fields__:
        if name in skip:
            continue
        value = getattr(options, name)
        payload[name] = value
    return payload


def run_batch(
    *,
    input_dir: Path,
    output_dir: Path,
    options_template: ConversionOptions,
    recursive: bool = False,
    fail_fast: bool = False,
    jobs: int = 1,
    skip_existing: bool = False,
    resume: bool = False,
    measure_quality: bool = False,
    quiet: bool = False,
    progress_stream: TextIO | None = None,
    convert_fn: ConvertFn = convert_pdf,
) -> tuple[BatchSummary, int]:
    """Discover PDFs, convert with failure isolation, write summaries.

    Returns ``(summary, exit_code)`` where exit_code is ``0`` on full success
    (including skips) and ``1`` if any file failed.
    """
    if jobs < 1:
        raise InputValidationError("--jobs must be >= 1")
    if jobs > 1:
        logger.warning(
            "Batch --jobs=%s uses processes; PaddleOCR is memory-heavy and parallel "
            "runs have OOM'd 16 GB machines. Prefer --jobs 1 unless you have ample RAM.",
            jobs,
        )

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs(input_dir, recursive=recursive)
    planned = plan_batch_outputs(
        pdfs,
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
    )

    summary_json_path = output_dir / SUMMARY_JSON_NAME
    resume_ok = load_resume_ok_files(summary_json_path) if resume else set()

    results: list[BatchFileResult] = []
    to_convert: list[DiscoveredPdf] = []
    for item in planned:
        if resume and item.relative_key in resume_ok:
            results.append(
                BatchFileResult(
                    file=item.relative_key,
                    output=_relative_or_str(item.output_path),
                    pages=None,
                    status="skipped",
                    seconds=0.0,
                    ocr_used=False,
                    ocr_pages=0,
                    raster_fallback_pages=0,
                    warnings_count=0,
                    planner_cost_usd=None,
                    quality={},
                    error="resume: already ok in summary.json",
                )
            )
            _emit_progress(
                progress_stream,
                quiet,
                f"skip (resume) {item.relative_key}",
            )
            continue
        if skip_existing and item.output_path.is_file():
            results.append(
                BatchFileResult(
                    file=item.relative_key,
                    output=_relative_or_str(item.output_path),
                    pages=None,
                    status="skipped",
                    seconds=0.0,
                    ocr_used=False,
                    ocr_pages=0,
                    raster_fallback_pages=0,
                    warnings_count=0,
                    planner_cost_usd=None,
                    quality={},
                    error="skip-existing: output pptx present",
                )
            )
            _emit_progress(
                progress_stream,
                quiet,
                f"skip (existing) {item.relative_key}",
            )
            continue
        to_convert.append(item)

    def options_factory(input_path: Path, output_path: Path, report_path: Path) -> ConversionOptions:
        kwargs = conversion_options_kwargs(options_template)
        # Per-file debug dir nesting when a shared debug root was requested.
        debug_dir = kwargs.get("debug_dir")
        if isinstance(debug_dir, Path):
            kwargs["debug_dir"] = debug_dir / input_path.stem
        refine_report = kwargs.get("refine_report_path")
        if refine_report is None and bool(kwargs.get("refine")):
            kwargs["refine_report_path"] = output_path.with_suffix(".refine.json")
        return ConversionOptions(
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            **kwargs,
        )

    failed = False
    total = len(planned)
    completed_base = len(results)

    if jobs == 1 or len(to_convert) <= 1:
        for index, item in enumerate(to_convert, start=1):
            _emit_progress(
                progress_stream,
                quiet,
                f"[{completed_base + index}/{total}] converting {item.relative_key} ...",
            )
            result = convert_one_planned(
                item,
                options_factory=options_factory,
                convert_fn=convert_fn,
                measure_quality=measure_quality,
            )
            results.append(result)
            _emit_progress(
                progress_stream,
                quiet,
                _format_result_line(result),
            )
            if result.status == "failed":
                failed = True
                if fail_fast:
                    break
    else:
        options_kwargs = conversion_options_kwargs(options_template)
        futures = {}
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for item in to_convert:
                payload = {
                    "input_path": str(item.input_path),
                    "relative_key": item.relative_key,
                    "output_path": str(item.output_path),
                    "options_kwargs": options_kwargs,
                    "measure_quality": measure_quality,
                }
                futures[executor.submit(_worker_convert, payload)] = item
            done_count = 0
            for future in as_completed(futures):
                item = futures[future]
                done_count += 1
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001
                    result = BatchFileResult(
                        file=item.relative_key,
                        output=_relative_or_str(item.output_path),
                        pages=None,
                        status="failed",
                        seconds=0.0,
                        ocr_used=False,
                        ocr_pages=0,
                        raster_fallback_pages=0,
                        warnings_count=0,
                        planner_cost_usd=None,
                        quality={},
                        error=str(error),
                    )
                results.append(result)
                _emit_progress(
                    progress_stream,
                    quiet,
                    f"[{completed_base + done_count}/{total}] {_format_result_line(result)}",
                )
                if result.status == "failed":
                    failed = True
                    if fail_fast:
                        for pending in futures:
                            pending.cancel()
                        break

    # Preserve discovery order in the summary.
    order = {item.relative_key: index for index, item in enumerate(planned)}
    results.sort(key=lambda row: order.get(row.file, 10**9))

    totals = compute_totals(results)
    summary = BatchSummary(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        results=results,
        totals=totals,
    )
    write_batch_summaries(summary, output_dir)
    exit_code = EXIT_PARTIAL_FAILURE if failed else EXIT_SUCCESS
    return summary, exit_code


def _emit_progress(stream: TextIO | None, quiet: bool, message: str) -> None:
    if quiet or stream is None:
        return
    stream.write(message + "\n")
    stream.flush()


def _format_result_line(result: BatchFileResult) -> str:
    if result.status == "ok":
        return f"ok {result.file} ({result.seconds:.3f}s, {result.pages} pages)"
    if result.status == "skipped":
        return f"skipped {result.file}"
    return f"FAILED {result.file}: {result.error}"


def compute_totals(results: Sequence[BatchFileResult]) -> dict[str, Any]:
    ok = sum(1 for row in results if row.status == "ok")
    failed = sum(1 for row in results if row.status == "failed")
    skipped = sum(1 for row in results if row.status == "skipped")
    pages = sum(row.pages or 0 for row in results if row.status == "ok")
    seconds = sum(row.seconds for row in results)
    ocr_pages = sum(row.ocr_pages for row in results)
    raster_pages = sum(row.raster_fallback_pages for row in results)
    warnings = sum(row.warnings_count for row in results)
    planner_costs = [row.planner_cost_usd for row in results if row.planner_cost_usd is not None]
    return {
        "files": len(results),
        "ok": ok,
        "failed": failed,
        "skipped": skipped,
        "pages": pages,
        "seconds": round(seconds, 3),
        "ocr_pages": ocr_pages,
        "raster_fallback_pages": raster_pages,
        "warnings_count": warnings,
        "planner_cost_usd": round(sum(planner_costs), 6) if planner_costs else None,
    }


def write_batch_summaries(summary: BatchSummary, output_dir: Path) -> None:
    """Write summary.json, summary.csv, and summary.md into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / SUMMARY_JSON_NAME
    json_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(summary, output_dir / SUMMARY_CSV_NAME)
    md = format_summary_markdown(summary)
    (output_dir / SUMMARY_MD_NAME).write_text(md + "\n", encoding="utf-8")


def _write_summary_csv(summary: BatchSummary, path: Path) -> None:
    fieldnames = [
        "file",
        "output",
        "pages",
        "status",
        "seconds",
        "ocr_used",
        "ocr_pages",
        "raster_fallback_pages",
        "warnings_count",
        "planner_cost_usd",
        "quality_score",
        "raster_fallback_fraction",
        "mean_editable_ratio",
        "mean_text_confidence",
        "ssim_mean",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.results:
            quality = row.quality or {}
            writer.writerow(
                {
                    "file": row.file,
                    "output": row.output or "",
                    "pages": "" if row.pages is None else row.pages,
                    "status": row.status,
                    "seconds": row.seconds,
                    "ocr_used": row.ocr_used,
                    "ocr_pages": row.ocr_pages,
                    "raster_fallback_pages": row.raster_fallback_pages,
                    "warnings_count": row.warnings_count,
                    "planner_cost_usd": "" if row.planner_cost_usd is None else row.planner_cost_usd,
                    "quality_score": quality.get("score", ""),
                    "raster_fallback_fraction": quality.get("raster_fallback_fraction", ""),
                    "mean_editable_ratio": quality.get("mean_editable_ratio", ""),
                    "mean_text_confidence": quality.get("mean_text_confidence", ""),
                    "ssim_mean": "" if row.ssim_mean is None else row.ssim_mean,
                    "error": row.error or "",
                }
            )


def format_summary_markdown(summary: BatchSummary) -> str:
    """Render a markdown table plus totals for stdout / summary.md."""
    headers = [
        "file",
        "pages",
        "status",
        "seconds",
        "ocr_pages",
        "raster_pages",
        "warnings",
        "quality",
        "planner_usd",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in summary.results:
        quality = row.quality or {}
        quality_cell = quality.get("score", "")
        planner = "" if row.planner_cost_usd is None else f"{row.planner_cost_usd:.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.file,
                    "" if row.pages is None else str(row.pages),
                    row.status,
                    f"{row.seconds:.3f}",
                    str(row.ocr_pages),
                    str(row.raster_fallback_pages),
                    str(row.warnings_count),
                    "" if quality_cell == "" else str(quality_cell),
                    planner,
                ]
            )
            + " |"
        )
    totals = summary.totals
    lines.append("")
    lines.append(
        f"**Totals:** files={totals.get('files', 0)} "
        f"ok={totals.get('ok', 0)} failed={totals.get('failed', 0)} "
        f"skipped={totals.get('skipped', 0)} pages={totals.get('pages', 0)} "
        f"seconds={totals.get('seconds', 0)}"
    )
    return "\n".join(lines)
