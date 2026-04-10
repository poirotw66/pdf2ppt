from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(slots=True)
class ConversionOptions:
    input_path: Path
    output_path: Path
    report_path: Path
    mode: str = "editable"
    lang: str = "ch"
    ocr_model_root: Path | None = Path("model")
    ocr_use_textline_orientation: bool = False
    ocr_det_thresh: float | None = None
    ocr_det_box_thresh: float | None = None
    ocr_drop_score: float | None = None
    render_dpi: int = 144
    background_dpi: int = 110
    background_image_format: str = "jpeg"
    background_jpeg_quality: int = 82
    debug_dir: Path | None = None
    use_doc_unwarping: bool = False
    inpaint_engine: str = "auto"
    inpaint_padding_px: int = 6
    inpaint_max_area_ratio: float = 0.12
    diffusion_command: str = "iopaint"
    diffusion_model: str = "brushnet"
    diffusion_device: str = "cuda"
    diffusion_max_crop_edge: int = 1024
    diffusion_complexity_threshold: float = 0.3
    diffusion_timeout_sec: float = 120.0


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
    pass


class OcrProcessingError(RuntimeError):
    pass
