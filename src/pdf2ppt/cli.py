from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from .core import DEFAULT_OCR_BATCH_SIZE
from .inpainting_engines import base_lama_inpaint_engine, uses_lama_patch_hybrid
from .pipeline import ConversionOptions, convert_pdf

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
            "textured surroundings and skipped when the outside ring is highly "
            "textured. Set negative to disable. Flat rings skip the API entirely."
        ),
    )
    parser.add_argument(
        "--generative-flat-ring-luma-std",
        type=float,
        default=4.0,
        help=(
            "When the outside-mask ring luma std is at or below this value, skip the "
            "generative API and use opencv-fast (smooth backgrounds)."
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
        default=16.0,
        help=(
            "Outside-ring luma std at/above this disables the low-frequency MAE gate "
            "(photo / textured pages where generative is preferred)."
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
        generative_flat_ring_luma_std=args.generative_flat_ring_luma_std,
        generative_textured_ring_luma_std=args.generative_textured_ring_luma_std,
        generative_safety_block_retries=args.generative_safety_block_retries,
        generative_max_attempts=args.generative_max_attempts,
    )
    report = convert_pdf(options, progress_callback=build_progress_callback(sys.stderr))
    print(
        f"Converted {report.input_path} -> {report.output_path} "
        f"({len(report.pages)} pages, report: {report_path})"
    )
    return 0
