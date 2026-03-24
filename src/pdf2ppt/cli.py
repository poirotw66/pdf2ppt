from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ConversionOptions, convert_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PDF files into editable PPTX decks.")
    parser.add_argument("input_pdf", type=Path, help="Input PDF file path.")
    parser.add_argument("output_pptx", type=Path, help="Output PPTX file path.")
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to write the JSON conversion report. Defaults next to the PPTX output.",
    )
    parser.add_argument(
        "--mode",
        choices=("editable", "fidelity", "fast"),
        default="editable",
        help="Conversion preset for rendering quality.",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="PaddleOCR language code. Defaults to 'ch' for Traditional/Chinese-heavy documents.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=144,
        help="Rendering DPI for preview/background generation.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Write per-page PNG debug overlays for OCR detection and masking alignment.",
    )
    parser.add_argument(
        "--enable-doc-unwarping",
        action="store_true",
        help="Enable PaddleOCR UVDoc unwarping. Disabled by default because it can distort slide layouts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report_path = args.report or args.output_pptx.with_suffix(".report.json")
    options = ConversionOptions(
        input_path=args.input_pdf,
        output_path=args.output_pptx,
        report_path=report_path,
        mode=args.mode,
        lang=args.lang,
        render_dpi=args.dpi,
        debug_dir=args.debug_dir or args.output_pptx.with_suffix("").with_name(f"{args.output_pptx.stem}_debug"),
        use_doc_unwarping=args.enable_doc_unwarping,
    )
    report = convert_pdf(options)
    print(
        f"Converted {report.input_path} -> {report.output_path} "
        f"({len(report.pages)} pages, report: {report_path})"
    )
    return 0
