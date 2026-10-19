from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_OCR_BATCH_SIZE = 3


class Pdf2PptError(RuntimeError):
    error_code = "pdf2ppt-error"

    def to_detail(self) -> dict[str, object]:
        return {
            "code": self.error_code,
            "message": str(self),
        }


class InputValidationError(Pdf2PptError):
    error_code = "input-error"


class PageConversionError(Pdf2PptError):
    error_code = "page-conversion-error"

    def __init__(self, page_number: int, message: str) -> None:
        self.page_number = page_number
        super().__init__(message)

    def to_detail(self) -> dict[str, object]:
        detail = super().to_detail()
        detail["page"] = self.page_number
        return detail


@dataclass(slots=True)
class ConversionOptions:
    input_path: Path
    output_path: Path
    report_path: Path
    mode: str = "editable"
    # Digital-PDF vector / table nativization. ``off`` keeps the pre-P1 path
    # (text boxes + raster background). ``auto`` (default) converts simple
    # drawings and detected tables when path complexity stays under the
    # threshold. ``aggressive`` raises the threshold and keeps denser paths.
    vector_mode: str = "auto"
    # Max drawings considered for vector nativization in ``auto`` mode.
    # ``aggressive`` uses ``max(threshold, 400)`` unless the caller sets higher.
    vector_complexity_threshold: int = 80
    # P2 layout analysis. Default ``heuristic`` keeps CPU/CI paths free of model
    # downloads. ``ppstructure`` opts into PP-StructureV3 / LayoutDetection and
    # soft-falls back to heuristic when the model is unavailable.
    layout_engine: str = "heuristic"
    layout_model_root: Path | None = Path("model")
    layout_skip_header_footer: bool = True
    lang: str = "ch"
    ocr_model_root: Path | None = Path("model")
    ocr_use_doc_orientation: bool = False
    ocr_use_textline_orientation: bool = False
    ocr_det_thresh: float | None = None
    ocr_det_box_thresh: float | None = None
    ocr_drop_score: float | None = None
    ocr_return_word_box: bool = False
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE
    render_dpi: int = 144
    background_dpi: int = 110
    background_image_format: str = "jpeg"
    background_jpeg_quality: int = 82
    debug_dir: Path | None = None
    use_doc_unwarping: bool = False
    inpaint_engine: str = "opencv-fast"
    inpaint_padding_px: int = 6
    # Absolute floor for mask dilation (px). Combined with line-height-scaled
    # dilation below so AA glyph fringes stay covered for all engines.
    inpaint_mask_dilate_px: int = 2
    # Dilate each text box by this fraction of its line height (clamped).
    inpaint_mask_dilate_line_height_ratio: float = 0.18
    inpaint_mask_dilate_min_px: int = 2
    inpaint_mask_dilate_max_px: int = 16
    inpaint_max_area_ratio: float = 0.12
    inpaint_model_root: Path | None = Path("model/lama")
    inpaint_onnx_cuda_provider: str = "CUDAExecutionProvider"
    inpaint_onnx_execution_mode: str = "sequential"
    inpaint_max_side_px: int = 1536
    inpaint_lama_repo_root: Path | None = Path("lama")
    inpaint_lama_device: str = "cuda"
    inpaint_lama_python_executable: Path | None = None
    approved_ocr_blocks_by_page: dict[int, list[Any]] = field(default_factory=dict)
    approved_ocr_image_size_by_page: dict[int, tuple[int, int]] = field(default_factory=dict)
    apply_notebooklm_watermark_fallback: bool = True
    inpaint_lama_patch_hybrid: bool = True
    # Opt-in generative (cloud) image-edit backend. Defaults keep behaviour identical
    # when no API key is present and when inpaint_engine is not generative-*.
    generative_model: str | None = None
    generative_api_key: str | None = None
    generative_max_calls: int = 50
    generative_max_cost_usd: float = 5.0
    generative_cache_dir: Path | None = Path(".pdf2ppt_cache/generative_inpaint")
    generative_timeout_s: float = 120.0
    generative_verify_ocr: bool = True
    # Reject generative patches whose low-frequency content differs from the
    # local opencv-fast result by more than this masked MAE (0–255). Set <0 to disable.
    # On textured surroundings the threshold is scaled up / skipped.
    generative_low_freq_mae_threshold: float = 4.0
    # When the outside ring texture score is this flat, skip the generative API entirely.
    generative_skip_flat_ring: bool = True
    # Measure for ring flat/textured routing: highpass (default), plane_residual, luma_std.
    generative_ring_texture_measure: str = "highpass"
    generative_ring_texture_blur_sigma: float = 5.0
    # Thresholds apply to the selected measure (historical CLI names keep "luma_std").
    generative_flat_ring_luma_std: float = 2.0
    # Ring texture score at/above this disables the low-freq gate (photo textures).
    generative_textured_ring_luma_std: float = 4.0
    # Retry once on nondeterministic copyright/recitation blocks before local fallback.
    generative_safety_block_retries: int = 1
    # Cap on total provider attempts including free safety blocks (None → 3× max_calls).
    generative_max_attempts: int | None = None
    generative_budget: Any | None = field(default=None, repr=False)
    generative_residual_checker: Any | None = field(default=None, repr=False)


@dataclass(slots=True)
class PageSignals:
    native_char_count: int
    native_text_area_ratio: float
    image_area_ratio: float
    drawing_count: int


@dataclass(slots=True)
class OcrPageData:
    blocks: list
    image: Image.Image


class OcrInitializationError(RuntimeError):
    error_code = "ocr-initialization-error"

    def to_detail(self) -> dict[str, object]:
        return {
            "code": self.error_code,
            "message": str(self),
        }


class OcrProcessingError(RuntimeError):
    error_code = "ocr-processing-error"

    def to_detail(self) -> dict[str, object]:
        return {
            "code": self.error_code,
            "message": str(self),
        }
