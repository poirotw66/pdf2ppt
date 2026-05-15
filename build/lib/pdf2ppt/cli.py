from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, TextIO

from .core import DEFAULT_OCR_BATCH_SIZE
from .pipeline import ConversionOptions, convert_pdf


def format_progress_line(completed: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return "Converting pages [------------------------] 0/0 (  0%)"
    safe_completed = min(max(completed, 0), total)
    filled = int(round((safe_completed / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    percent = int(round((safe_completed / total) * 100))
    return f"Converting pages [{bar}] {safe_completed}/{total} ({percent:>3}%)"


def build_progress_callback(
    stream: TextIO,
    *,
    width: int = 24,
) -> Callable[[int, int], None]:
    last_line: str | None = None
    interactive = bool(getattr(stream, "isatty", lambda: False)())

    def callback(completed: int, total: int) -> None:
        nonlocal last_line
        line = format_progress_line(completed, total, width=width)
        if interactive:
            stream.write(f"\r{line}")
            if total > 0 and completed >= total:
                stream.write("\n")
        elif line != last_line:
            stream.write(f"{line}\n")
        stream.flush()
        last_line = line

    return callback


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


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
        "--ocr-model-root",
        type=Path,
        default=Path("model"),
        help="Directory used for local PaddleOCR model storage. Defaults to ./model.",
    )
    parser.add_argument(
        "--enable-doc-orientation",
        action="store_true",
        help="Enable PaddleOCR document orientation classification. Disabled by default to reduce startup and OCR latency.",
    )
    parser.add_argument(
        "--enable-textline-orientation",
        action="store_true",
        help="Enable PaddleOCR textline orientation classification. Disabled by default to reduce startup and OCR latency.",
    )
    parser.add_argument(
        "--ocr-det-thresh",
        type=float,
        help="PaddleOCR text detection threshold. Omit to use the PaddleOCR default.",
    )
    parser.add_argument(
        "--ocr-det-box-thresh",
        type=float,
        help="PaddleOCR text detection box threshold. Omit to use the PaddleOCR default.",
    )
    parser.add_argument(
        "--ocr-drop-score",
        type=float,
        help="PaddleOCR recognition score threshold. Omit to use the PaddleOCR default.",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=positive_int,
        default=DEFAULT_OCR_BATCH_SIZE,
        help="Number of pages processed together for full-page OCR. Defaults to 3.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=144,
        help="Rendering DPI for OCR-oriented page rasterization.",
    )
    parser.add_argument(
        "--background-dpi",
        type=int,
        default=110,
        help="Rendering DPI for full-page or overlay background images embedded into the PPTX.",
    )
    parser.add_argument(
        "--background-format",
        choices=("jpeg", "png"),
        default="jpeg",
        help="Image format used for embedded background pages. JPEG is smaller; PNG preserves lossless detail.",
    )
    parser.add_argument(
        "--background-jpeg-quality",
        type=int,
        default=82,
        help="JPEG quality for embedded background pages when --background-format=jpeg.",
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
    parser.add_argument(
        "--inpaint-engine",
        choices=("auto", "white-box", "opencv-fast", "lama-onnx-cuda"),
        default="opencv-fast",
        help="Background reconstruction engine for overlay pages.",
    )
    parser.add_argument(
        "--inpaint-padding-px",
        type=int,
        default=6,
        help="Extra mask dilation padding in pixels before background reconstruction.",
    )
    parser.add_argument(
        "--inpaint-max-area-ratio",
        type=float,
        default=0.12,
        help="Fallback to white-box masking when the overlay mask covers more than this fraction of the page.",
    )
    parser.add_argument(
        "--inpaint-model-root",
        type=Path,
        default=Path("model/lama"),
        help="Directory or .onnx file for the optional lama-onnx-cuda background model.",
    )
    parser.add_argument(
        "--inpaint-onnx-cuda-provider",
        default="CUDAExecutionProvider",
        help="ONNX Runtime provider name used by --inpaint-engine lama-onnx-cuda.",
    )
    parser.add_argument(
        "--inpaint-onnx-execution-mode",
        choices=("sequential", "parallel"),
        default="sequential",
        help="ONNX Runtime execution mode used by --inpaint-engine lama-onnx-cuda.",
    )
    parser.add_argument(
        "--inpaint-max-side-px",
        type=positive_int,
        default=1536,
        help="Maximum image side passed into lama-onnx-cuda before proportional downscaling.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity for conversion diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    report_path = args.report or args.output_pptx.with_suffix(".report.json")
    options = ConversionOptions(
        input_path=args.input_pdf,
        output_path=args.output_pptx,
        report_path=report_path,
        mode=args.mode,
        lang=args.lang,
        ocr_model_root=args.ocr_model_root,
        ocr_use_doc_orientation=args.enable_doc_orientation,
        ocr_use_textline_orientation=args.enable_textline_orientation,
        ocr_det_thresh=args.ocr_det_thresh,
        ocr_det_box_thresh=args.ocr_det_box_thresh,
        ocr_drop_score=args.ocr_drop_score,
        ocr_batch_size=args.ocr_batch_size,
        render_dpi=args.dpi,
        background_dpi=args.background_dpi,
        background_image_format=args.background_format,
        background_jpeg_quality=args.background_jpeg_quality,
        debug_dir=args.debug_dir,
        use_doc_unwarping=args.enable_doc_unwarping,
        inpaint_engine=args.inpaint_engine,
        inpaint_padding_px=args.inpaint_padding_px,
        inpaint_max_area_ratio=args.inpaint_max_area_ratio,
        inpaint_model_root=args.inpaint_model_root,
        inpaint_onnx_cuda_provider=args.inpaint_onnx_cuda_provider,
        inpaint_onnx_execution_mode=args.inpaint_onnx_execution_mode,
        inpaint_max_side_px=args.inpaint_max_side_px,
    )
    report = convert_pdf(options, progress_callback=build_progress_callback(sys.stderr))
    print(
        f"Converted {report.input_path} -> {report.output_path} "
        f"({len(report.pages)} pages, report: {report_path})"
    )
    return 0
