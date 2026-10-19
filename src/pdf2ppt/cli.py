from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from .core import DEFAULT_OCR_BATCH_SIZE, ConversionOptions, InputValidationError
from .inpainting_engines import base_lama_inpaint_engine, uses_lama_patch_hybrid
from .pipeline import convert_from_ir, convert_pdf

GENERATIVE_INPAINT_ENGINE_CHOICES = (
    "generative-openai",
    "generative-gemini",
)


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
    parser = argparse.ArgumentParser(
        description=(
            "Convert PDF files into editable PPTX decks, or round-trip via SlideIR "
            "(--emit-ir / --from-ir). Subcommands: emit-ir, from-ir, convert."
        )
    )
    parser.add_argument(
        "input_pdf",
        type=Path,
        nargs="?",
        help="Input PDF file path (omit when using --from-ir).",
    )
    parser.add_argument(
        "output_pptx",
        type=Path,
        nargs="?",
        help="Output PPTX file path (omit with --emit-ir --ir-only).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="output_override",
        help="Explicit output PPTX path (useful with --from-ir).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to write the JSON conversion report. Defaults next to the PPTX or IR output.",
    )
    parser.add_argument(
        "--emit-ir",
        type=Path,
        help=(
            "Write SlideIR JSON (plus a sibling {ir_filename}.assets/ directory for "
            "backgrounds and cropped images). Can be combined with PPTX output, or used "
            "alone with --ir-only."
        ),
    )
    parser.add_argument(
        "--ir-only",
        action="store_true",
        help="With --emit-ir, skip PPTX generation (analysis + SlideIR only).",
    )
    parser.add_argument(
        "--from-ir",
        type=Path,
        help=(
            "Load SlideIR JSON and render PPTX (no PDF). Assets are resolved from the "
            "assets_dir next to the IR file. Output path: positional or -o/--output."
        ),
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
        "--ocr-return-word-box",
        action="store_true",
        help=(
            "Return PaddleOCR word-level boxes. Disabled by default so each detected text line "
            "stays a single OCR block after line merging."
        ),
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
        choices=(
            "auto",
            "white-box",
            "opencv-fast",
            "lama-onnx",
            "lama-onnx-cuda",
            "lama-pytorch",
            "lama-onnx-hybrid",
            "lama-onnx-cuda-hybrid",
            "lama-pytorch-hybrid",
            *GENERATIVE_INPAINT_ENGINE_CHOICES,
        ),
        default="opencv-fast",
        help=(
            "Background reconstruction engine for overlay pages. 'lama-onnx-cuda' and "
            "'lama-onnx-cuda-hybrid' are backward-compatible aliases for 'lama-onnx' and "
            "'lama-onnx-hybrid'; the ONNX engine runs on CPU when no CUDA provider is available. "
            "generative-openai / generative-gemini are opt-in cloud image-edit backends "
            "(require API keys; never selected by auto)."
        ),
    )
    parser.add_argument(
        "--inpaint-padding-px",
        type=int,
        default=6,
        help="Extra mask dilation padding in pixels before background reconstruction.",
    )
    parser.add_argument(
        "--inpaint-mask-dilate-px",
        type=int,
        default=2,
        help=(
            "Absolute minimum dilation (px) applied to each text box after "
            "line-height-scaled expansion. Default 2; set 0 to rely only on the "
            "line-height ratio. Applies to all inpaint engines."
        ),
    )
    parser.add_argument(
        "--inpaint-mask-dilate-line-height-ratio",
        type=float,
        default=0.18,
        help=(
            "Dilate each text mask by this fraction of the box line height "
            "(default 0.18 ≈ 18%%). Clamped by --inpaint-mask-dilate-min/max-px."
        ),
    )
    parser.add_argument(
        "--inpaint-mask-dilate-min-px",
        type=int,
        default=2,
        help="Minimum per-box dilation in pixels when line-height scaling is enabled.",
    )
    parser.add_argument(
        "--inpaint-mask-dilate-max-px",
        type=int,
        default=16,
        help="Maximum per-box dilation in pixels when line-height scaling is enabled.",
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
        help="Directory or .onnx file for the optional lama-onnx background model.",
    )
    parser.add_argument(
        "--inpaint-onnx-cuda-provider",
        default="CUDAExecutionProvider",
        help=(
            "ONNX Runtime provider name used by --inpaint-engine lama-onnx. Falls back to "
            "CPUExecutionProvider automatically when this provider is unavailable."
        ),
    )
    parser.add_argument(
        "--inpaint-onnx-execution-mode",
        choices=("sequential", "parallel"),
        default="sequential",
        help="ONNX Runtime execution mode used by --inpaint-engine lama-onnx.",
    )
    parser.add_argument(
        "--inpaint-max-side-px",
        type=positive_int,
        default=1536,
        help="Maximum image side passed into lama-onnx before proportional downscaling.",
    )
    parser.add_argument(
        "--inpaint-lama-repo-root",
        type=Path,
        default=Path("lama"),
        help="Path to the official advimman/lama repository root used by --inpaint-engine lama-pytorch.",
    )
    parser.add_argument(
        "--inpaint-lama-device",
        default="cuda",
        help="Device override passed to the official advimman/lama predictor when --inpaint-engine lama-pytorch.",
    )
    parser.add_argument(
        "--inpaint-lama-python",
        type=Path,
        help=(
            "Python executable for the official advimman/lama repo when using --inpaint-engine lama-pytorch. "
            "Defaults to PDF2PPT_LAMA_PYTHON, a conda env named lama, or the current interpreter."
        ),
    )
    parser.add_argument(
        "--generative-model",
        default=None,
        help=(
            "Optional model id override for generative-openai / generative-gemini. "
            "Defaults: gpt-image-2 (OpenAI) or gemini-nano-banana-2.1 (Gemini). "
            "Also overridable via PDF2PPT_OPENAI_IMAGE_MODEL / PDF2PPT_GEMINI_IMAGE_MODEL."
        ),
    )
    parser.add_argument(
        "--generative-max-calls",
        type=int,
        default=50,
        help="Per-run maximum number of paid generative API calls (cost brake).",
    )
    parser.add_argument(
        "--generative-max-cost-usd",
        type=float,
        default=5.0,
        help="Per-run estimated USD budget for generative API calls (cost brake).",
    )
    parser.add_argument(
        "--generative-cache-dir",
        type=Path,
        default=Path(".pdf2ppt_cache/generative_inpaint"),
        help="Disk cache directory for generative patch results (keyed by image+mask+model hash).",
    )
    parser.add_argument(
        "--generative-timeout-s",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for each generative API call.",
    )
    parser.add_argument(
        "--no-generative-verify-ocr",
        action="store_true",
        help="Disable the post-generation OCR residual-text safety check (not recommended).",
    )
    parser.add_argument(
        "--generative-low-freq-mae-threshold",
        type=float,
        default=4.0,
        help=(
            "Base low-frequency MAE gate vs opencv-fast (0-255). Scaled up on "
            "textured surroundings (per --generative-ring-texture-measure) and "
            "skipped when the outside ring is highly textured. Set negative to "
            "disable. Flat rings skip the API entirely."
        ),
    )
    parser.add_argument(
        "--generative-ring-texture-measure",
        choices=["highpass", "plane_residual", "luma_std"],
        default="highpass",
        help=(
            "How to score outside-mask ring texture for flat-API-skip and low-freq "
            "gate scaling. Default highpass (ring minus Gaussian blur) treats smooth "
            "gradients as flat; plane_residual fits a plane; luma_std is raw brightness "
            "variation (legacy, mis-labels full-page gradients as textured)."
        ),
    )
    parser.add_argument(
        "--generative-ring-texture-blur-sigma",
        type=float,
        default=5.0,
        help="Gaussian sigma for the highpass ring texture measure (ignored otherwise).",
    )
    parser.add_argument(
        "--generative-flat-ring-luma-std",
        type=float,
        default=2.0,
        help=(
            "When the outside-mask ring texture score is at or below this value, skip "
            "the generative API and use opencv-fast (smooth / gradient backgrounds). "
            "Threshold applies to --generative-ring-texture-measure."
        ),
    )
    parser.add_argument(
        "--no-generative-skip-flat-ring",
        action="store_true",
        help="Always call the generative API even when the outside ring is flat.",
    )
    parser.add_argument(
        "--generative-textured-ring-luma-std",
        type=float,
        default=4.0,
        help=(
            "Outside-ring texture score at/above this disables the low-frequency MAE "
            "gate (photo / textured pages where generative is preferred). Threshold "
            "applies to --generative-ring-texture-measure (default highpass ≈ 4)."
        ),
    )
    parser.add_argument(
        "--generative-safety-block-retries",
        type=int,
        default=1,
        help=(
            "Retry count for nondeterministic copyright/recitation (and similar) "
            "safety blocks before falling back to opencv-fast. Blocked attempts are free "
            "and do not consume --generative-max-calls."
        ),
    )
    parser.add_argument(
        "--generative-max-attempts",
        type=int,
        default=None,
        help=(
            "Hard cap on total provider attempts including free safety blocks. "
            "Defaults to 3× --generative-max-calls."
        ),
    )
    parser.add_argument(
        "--vector-mode",
        choices=("off", "auto", "aggressive"),
        default="auto",
        help=(
            "Digital-PDF vector/table nativization. 'auto' (default) maps simple drawings "
            "and find_tables() grids to native PowerPoint shapes/tables when complexity "
            "stays under --vector-complexity-threshold; 'aggressive' raises the threshold; "
            "'off' keeps the pre-P1 text+raster path. Scanned pages are unchanged."
        ),
    )
    parser.add_argument(
        "--vector-complexity-threshold",
        type=positive_int,
        default=80,
        help=(
            "Max drawings considered for vector nativization in --vector-mode=auto "
            "(default 80). Aggressive mode uses max(threshold, 400)."
        ),
    )
    parser.add_argument(
        "--layout-engine",
        choices=("heuristic", "ppstructure"),
        default="heuristic",
        help=(
            "Layout analysis for text-box grouping, bullets, titles, figures, and reading "
            "order. 'heuristic' (default) uses geometric rules and needs no model download "
            "(CI/CPU-safe). 'ppstructure' opts into PaddleOCR PP-StructureV3 / LayoutDetection "
            "and soft-falls back to heuristic when the model is unavailable. See "
            "docs/layout-analysis.md."
        ),
    )
    parser.add_argument(
        "--layout-model-root",
        type=Path,
        default=Path("model"),
        help="Directory for PP-StructureV3 / DocLayout model storage. Defaults to ./model.",
    )
    parser.add_argument(
        "--keep-header-footer",
        action="store_true",
        help=(
            "Keep header/footer/page-number layout regions as text boxes. "
            "By default they are skipped for consistent slide cleanup."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity for conversion diagnostics.",
    )
    return parser


def _rewrite_subcommand_argv(argv: list[str]) -> list[str]:
    """Map ``emit-ir`` / ``from-ir`` / ``convert`` subcommands onto flag form."""
    if not argv:
        return argv
    command = argv[0]
    rest = argv[1:]
    if command == "convert":
        return rest
    if command == "emit-ir":
        # pdf2ppt emit-ir input.pdf out.slideir.json [flags...]
        if len(rest) < 2 or rest[0].startswith("-") or rest[1].startswith("-"):
            raise SystemExit(
                "usage: pdf2ppt emit-ir INPUT.pdf OUTPUT.slideir.json [options...]"
            )
        input_pdf, ir_path, *flags = rest
        return [input_pdf, "--emit-ir", ir_path, "--ir-only", *flags]
    if command == "from-ir":
        # pdf2ppt from-ir input.slideir.json output.pptx [flags...]
        if len(rest) < 2 or rest[0].startswith("-") or rest[1].startswith("-"):
            raise SystemExit("usage: pdf2ppt from-ir INPUT.slideir.json OUTPUT.pptx [options...]")
        ir_path, output_pptx, *flags = rest
        return ["--from-ir", ir_path, "-o", output_pptx, *flags]
    return argv


def _resolve_output_pptx(args: argparse.Namespace) -> Path | None:
    return args.output_override or args.output_pptx


def _build_conversion_options(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> ConversionOptions:
    return ConversionOptions(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        mode=args.mode,
        lang=args.lang,
        ocr_model_root=args.ocr_model_root,
        ocr_use_doc_orientation=args.enable_doc_orientation,
        ocr_use_textline_orientation=args.enable_textline_orientation,
        ocr_det_thresh=args.ocr_det_thresh,
        ocr_det_box_thresh=args.ocr_det_box_thresh,
        ocr_drop_score=args.ocr_drop_score,
        ocr_return_word_box=args.ocr_return_word_box,
        ocr_batch_size=args.ocr_batch_size,
        render_dpi=args.dpi,
        background_dpi=args.background_dpi,
        background_image_format=args.background_format,
        background_jpeg_quality=args.background_jpeg_quality,
        debug_dir=args.debug_dir,
        use_doc_unwarping=args.enable_doc_unwarping,
        inpaint_engine=base_lama_inpaint_engine(args.inpaint_engine),
        inpaint_lama_patch_hybrid=uses_lama_patch_hybrid(args.inpaint_engine),
        inpaint_padding_px=args.inpaint_padding_px,
        inpaint_mask_dilate_px=args.inpaint_mask_dilate_px,
        inpaint_mask_dilate_line_height_ratio=args.inpaint_mask_dilate_line_height_ratio,
        inpaint_mask_dilate_min_px=args.inpaint_mask_dilate_min_px,
        inpaint_mask_dilate_max_px=args.inpaint_mask_dilate_max_px,
        inpaint_max_area_ratio=args.inpaint_max_area_ratio,
        inpaint_model_root=args.inpaint_model_root,
        inpaint_onnx_cuda_provider=args.inpaint_onnx_cuda_provider,
        inpaint_onnx_execution_mode=args.inpaint_onnx_execution_mode,
        inpaint_max_side_px=args.inpaint_max_side_px,
        inpaint_lama_repo_root=args.inpaint_lama_repo_root,
        inpaint_lama_device=args.inpaint_lama_device,
        inpaint_lama_python_executable=args.inpaint_lama_python,
        generative_model=args.generative_model,
        generative_max_calls=args.generative_max_calls,
        generative_max_cost_usd=args.generative_max_cost_usd,
        generative_cache_dir=args.generative_cache_dir,
        generative_timeout_s=args.generative_timeout_s,
        generative_verify_ocr=not args.no_generative_verify_ocr,
        generative_low_freq_mae_threshold=args.generative_low_freq_mae_threshold,
        generative_skip_flat_ring=not args.no_generative_skip_flat_ring,
        generative_ring_texture_measure=args.generative_ring_texture_measure,
        generative_ring_texture_blur_sigma=args.generative_ring_texture_blur_sigma,
        generative_flat_ring_luma_std=args.generative_flat_ring_luma_std,
        generative_textured_ring_luma_std=args.generative_textured_ring_luma_std,
        generative_safety_block_retries=args.generative_safety_block_retries,
        generative_max_attempts=args.generative_max_attempts,
        vector_mode=args.vector_mode,
        vector_complexity_threshold=args.vector_complexity_threshold,
        layout_engine=args.layout_engine,
        layout_model_root=args.layout_model_root,
        layout_skip_header_footer=not args.keep_header_footer,
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        raw_argv = _rewrite_subcommand_argv(raw_argv)
    except SystemExit as error:
        message = str(error)
        if message:
            print(message, file=sys.stderr)
        return 2

    parser = build_parser()
    args = parser.parse_args(raw_argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.from_ir is not None:
            if args.emit_ir is not None or args.ir_only:
                raise InputValidationError("--from-ir cannot be combined with --emit-ir / --ir-only.")
            output_pptx = _resolve_output_pptx(args)
            # Allow ``pdf2ppt --from-ir deck.json out.pptx`` where out.pptx lands in input_pdf.
            if output_pptx is None and args.input_pdf is not None:
                output_pptx = args.input_pdf
            if output_pptx is None:
                raise InputValidationError(
                    "--from-ir requires an output PPTX path (positional or -o/--output)."
                )
            report_path = args.report or output_pptx.with_suffix(".report.json")
            report = convert_from_ir(args.from_ir, output_pptx, report_path=report_path)
            print(
                f"Rendered SlideIR {args.from_ir} -> {output_pptx} "
                f"({len(report.pages)} pages, report: {report_path})"
            )
            return 0

        if args.input_pdf is None:
            raise InputValidationError("Input PDF path is required unless --from-ir is set.")

        write_pptx = not args.ir_only
        if args.ir_only and args.emit_ir is None:
            raise InputValidationError("--ir-only requires --emit-ir PATH.")

        output_pptx = _resolve_output_pptx(args)
        if write_pptx and output_pptx is None:
            raise InputValidationError("Output PPTX path is required unless --ir-only is set.")
        if output_pptx is None:
            # IR-only: placeholder path for ConversionOptions (not written).
            output_pptx = args.emit_ir.with_suffix(".pptx") if args.emit_ir is not None else Path("output.pptx")

        report_path = args.report
        if report_path is None:
            if args.ir_only and args.emit_ir is not None:
                report_path = args.emit_ir.with_suffix(".report.json")
            else:
                report_path = output_pptx.with_suffix(".report.json")

        options = _build_conversion_options(
            args,
            input_path=args.input_pdf,
            output_path=output_pptx,
            report_path=report_path,
        )
        report = convert_pdf(
            options,
            progress_callback=build_progress_callback(sys.stderr),
            write_pptx=write_pptx,
            emit_ir_path=args.emit_ir,
        )
        parts = []
        if write_pptx:
            parts.append(str(options.output_path))
        if args.emit_ir is not None:
            parts.append(f"ir={args.emit_ir}")
        dest = ", ".join(parts) if parts else "(no output)"
        print(
            f"Converted {report.input_path} -> {dest} "
            f"({len(report.pages)} pages, report: {report_path})"
        )
        return 0
    except InputValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
