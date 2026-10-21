#!/usr/bin/env python3
"""Real LibreOffice refine budget bench on a synthetic ~30-page deck.

Requires soffice on PATH. Builds intentional shape-colour defects so refine has
work, then reports pages tried / accepts / SSIM / wall time for 10/30/90s.
"""

from __future__ import annotations

import argparse
import io
import tempfile
from pathlib import Path

import fitz
from PIL import Image

from pdf2ppt.models import PageResult, QualityScore, ShapeElement
from pdf2ppt.pipeline import render_pages_to_pptx
from pdf2ppt.refine import refine_pages
from pdf2ppt.render_compare import find_libreoffice_executable


def _page(index: int, *, dirty: bool) -> PageResult:
    shapes = []
    if dirty:
        shapes = [
            ShapeElement(
                id=f"card_{index}",
                kind="rect",
                bbox=(72, 72, 240, 180),
                fill_color="#FF0000",  # wrong vs blue reference
            )
        ]
    return PageResult(
        page_number=index + 1,
        page_kind="digital",
        background_mode="elements",
        width_pt=612.0,
        height_pt=792.0,
        text_blocks=[],
        quality_score=QualityScore(1.0, 1.0, 1.0, 1.0),
        fallback_reason=None,
        shape_elements=shapes,
    )


def _build_reference_pdf(path: Path, page_count: int) -> None:
    doc = fitz.open()
    for index in range(page_count):
        page = doc.new_page(width=612, height=792)
        # Blue rect matching the "correct" colour refine should recover toward.
        if index % 2 == 0 or index < 8:
            page.draw_rect(fitz.Rect(72, 72, 240, 180), color=(0, 0.4, 0.8), fill=(0, 0.4, 0.8))
        page.insert_text((72, 40), f"Page {index + 1}", fontsize=18)
    doc.save(path)
    doc.close()


def _render_pdf_page(pdf_path: Path, page_index: int, *, dpi: float = 72) -> Image.Image:
    with fitz.open(pdf_path) as document:
        page = document.load_page(page_index)
        scale = float(dpi) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def run_budget(budget: float, page_count: int = 30, dpi: float = 72.0) -> dict[str, object]:
    pages = [_page(i, dirty=(i % 2 == 0 or i < 8)) for i in range(page_count)]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf_path = root / "deck.pdf"
        pptx_path = root / "deck.pptx"
        _build_reference_pdf(pdf_path, page_count)
        render_pages_to_pptx(pages, pptx_path, slide_width_pt=612.0, slide_height_pt=792.0)

        def render_pdf(path, index, *, dpi=72):
            return _render_pdf_page(Path(path), index, dpi=dpi)

        result_pages, report = refine_pages(
            pages,
            pdf_path=pdf_path,
            output_pptx=pptx_path,
            slide_width_pt=612.0,
            slide_height_pt=792.0,
            iterations=2,
            max_fixes_per_slide=2,
            max_attempts_per_slide=8,
            time_budget_seconds=budget,
            dpi=dpi,
            render_pdf_page=render_pdf,
            work_dir=root / "refine",
        )

    tried = [p for p in report.pages if (p.attempt_count or 0) > 0]
    accepts = sum(p.accepted_fix_count for p in report.pages)
    befores = [p.metric_before for p in report.pages if p.metric_before is not None]
    afters = [p.metric_after for p in report.pages if p.metric_after is not None]
    mean_before = sum(befores) / len(befores) if befores else None
    mean_after = sum(afters) / len(afters) if afters else None
    used_sum = sum(float(p.time_used_seconds or 0.0) for p in report.pages)
    return {
        "budget": budget,
        "baseline_mode": report.baseline_mode,
        "baseline_page_count": report.baseline_page_count,
        "pages_skipped": len(report.pages_skipped),
        "pages_tried": len(tried),
        "accepts": accepts,
        "mean_ssim_before": None if mean_before is None else round(mean_before, 6),
        "mean_ssim_after": None if mean_after is None else round(mean_after, 6),
        "wall_s": report.runtime_seconds,
        "used_sum": round(used_sum, 3),
        "soffice_calls": report.soffice_call_count,
        "reallocation_passes": report.reallocation_passes,
        "overrun": round(max(0.0, float(report.runtime_seconds or 0.0) - budget), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=30)
    parser.add_argument("--dpi", type=float, default=72.0)
    args = parser.parse_args()
    if find_libreoffice_executable() is None:
        raise SystemExit("LibreOffice (soffice) is required for this bench")
    print(f"Real soffice refine bench ({args.pages} pages, dpi={args.dpi})")
    print(
        f"{'budget':>8} {'mode':>8} {'base_n':>6} {'skip':>5} {'tried':>5} "
        f"{'accept':>6} {'ssim_b':>10} {'ssim_a':>10} {'wall':>7} {'over':>6} "
        f"{'soffice':>7} {'realloc':>7}"
    )
    for budget in (10.0, 30.0, 90.0):
        row = run_budget(budget, page_count=args.pages, dpi=args.dpi)
        print(
            f"{row['budget']:>8} {str(row['baseline_mode']):>8} {row['baseline_page_count']:>6} "
            f"{row['pages_skipped']:>5} {row['pages_tried']:>5} {row['accepts']:>6} "
            f"{str(row['mean_ssim_before']):>10} {str(row['mean_ssim_after']):>10} "
            f"{row['wall_s']:>7} {row['overrun']:>6} {row['soffice_calls']:>7} "
            f"{row['reallocation_passes']:>7}"
        )


if __name__ == "__main__":
    main()
