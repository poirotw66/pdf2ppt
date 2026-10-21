#!/usr/bin/env python3
"""Synthetic 30-page refine budget bench (mock soffice timings).

Reports pages tried / accepts / mean SSIM before·after / wall time for budgets
10s / 30s / 90s. Used for PR measurement tables when LibreOffice is unavailable.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from pdf2ppt.models import PageResult, QualityScore, ShapeElement, TextBlock
from pdf2ppt.refine import refine_pages


def _page(index: int, *, dirty: bool) -> PageResult:
    shapes = []
    texts = []
    if dirty:
        shapes = [
            ShapeElement(
                id=f"card_{index}",
                kind="rect",
                bbox=(12, 12, 40, 40),
                fill_color="#FF0000",
            )
        ]
        texts = [
            TextBlock(
                id=f"t{index}",
                source="native",
                bbox=(50, 10, 120, 28),
                text=f"Title {index}",
                confidence=1.0,
                font_size=28.0 if index % 3 == 0 else 14.0,
            )
        ]
    return PageResult(
        page_number=index + 1,
        page_kind="digital",
        background_mode="elements",
        width_pt=64.0,
        height_pt=36.0,
        text_blocks=texts,
        quality_score=QualityScore(1.0, 1.0, 1.0, 1.0),
        fallback_reason=None,
        shape_elements=shapes,
    )


def _solid(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (64, 36), color=color)


def run_budget(budget: float, page_count: int = 30) -> dict[str, object]:
    # Mix of dirty / clean pages so severity and prerank have signal.
    pages = [_page(i, dirty=(i % 2 == 0 or i < 8)) for i in range(page_count)]
    reference = _solid((255, 255, 255))
    ImageDraw.Draw(reference).rectangle((12, 12, 40, 40), fill=(0, 100, 200))
    bad = _solid((255, 255, 255))
    ImageDraw.Draw(bad).rectangle((12, 12, 40, 40), fill=(200, 0, 0))

    written: dict[str, int] = {}
    digests = iter([f"d{i}" for i in range(5000)])

    def fake_write(pages_arg, output_path, *, slide_width_pt, slide_height_pt):
        path = Path(output_path)
        path.write_bytes(b"PK fake")
        written[path.name] = len(pages_arg)
        return path

    def fake_render(pptx_path, *, dpi=144, work_dir=None, timeout_seconds=180):
        name = Path(pptx_path).name
        n = written.get(name, 1)
        label = str(work_dir or "")
        if "baseline" in label or name in {"out.pptx", "baseline-partial.pptx"}:
            # Calibrated to PR #16: ~5s for 30 pages ≈ 1.5s fixed + 0.12s/page.
            time.sleep(min(float(timeout_seconds) - 0.05, 1.5 + 0.12 * n))
            return [bad.copy() for _ in range(n)]
        time.sleep(min(float(timeout_seconds) - 0.05, 1.6))
        # Accept often enough to show budget effects.
        if "trial" in label and hash(label) % 3 == 0:
            return [reference.copy()]
        return [bad.copy()]

    def fake_pdf(_path, _index, *, dpi=144):
        return reference.copy()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.pptx"
        (Path(tmp) / "src.pdf").write_bytes(b"%PDF-1.4")
        with patch("pdf2ppt.refine.find_libreoffice_executable", return_value=Path("/usr/bin/soffice")):
            with patch(
                "pdf2ppt.refine.pptx_content_digest",
                side_effect=lambda _p: next(digests, "done"),
            ):
                with patch("pdf2ppt.refine.slide_editability_score", return_value=0.5):
                    with patch("pdf2ppt.refine.slide_text_f1_vs_pdf", return_value=1.0):
                        _pages, report = refine_pages(
                            pages,
                            pdf_path=Path(tmp) / "src.pdf",
                            output_pptx=out,
                            slide_width_pt=64.0,
                            slide_height_pt=36.0,
                            iterations=2,
                            max_fixes_per_slide=2,
                            max_attempts_per_slide=8,
                            time_budget_seconds=budget,
                            dpi=72,
                            write_pptx=fake_write,
                            render_pptx=fake_render,
                            render_pdf_page=fake_pdf,
                            work_dir=Path(tmp) / "work",
                        )

    tried = [p for p in report.pages if (p.attempt_count or 0) > 0]
    accepts = sum(p.accepted_fix_count for p in report.pages)
    befores = [p.metric_before for p in report.pages if p.metric_before is not None]
    afters = [p.metric_after for p in report.pages if p.metric_after is not None]
    mean_before = sum(befores) / len(befores) if befores else None
    mean_after = sum(afters) / len(afters) if afters else None
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
        "soffice_calls": report.soffice_call_count,
        "reallocation_passes": report.reallocation_passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=30)
    args = parser.parse_args()
    print(f"Synthetic {args.pages}-page refine budget bench (mock soffice)")
    print(
        f"{'budget':>8} {'mode':>8} {'base_n':>6} {'skip':>5} {'tried':>5} "
        f"{'accept':>6} {'ssim_b':>10} {'ssim_a':>10} {'wall':>7} {'soffice':>7} {'realloc':>7}"
    )
    for budget in (10.0, 30.0, 90.0):
        row = run_budget(budget, page_count=args.pages)
        print(
            f"{row['budget']:>8} {str(row['baseline_mode']):>8} {row['baseline_page_count']:>6} "
            f"{row['pages_skipped']:>5} {row['pages_tried']:>5} {row['accepts']:>6} "
            f"{str(row['mean_ssim_before']):>10} {str(row['mean_ssim_after']):>10} "
            f"{row['wall_s']:>7} {row['soffice_calls']:>7} {row['reallocation_passes']:>7}"
        )


if __name__ == "__main__":
    main()
