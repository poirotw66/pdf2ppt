from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .core import DEFAULT_INPAINT_ENGINE, DEFAULT_OCR_BATCH_SIZE

DEFAULT_DETECT_CONFIDENCE_THRESHOLD = 0.75


class ErrorDetail(BaseModel):
    code: str
    message: str
    page: int | None = None


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class JobResponse(BaseModel):
    job_id: str
    status: str
    original_filename: str
    page_count: int
    created_at: str
    updated_at: str
    detection_path: str | None = None
    approved_boxes_path: str | None = None
    output_pptx_path: str | None = None
    report_path: str | None = None


class DetectRequest(BaseModel):
    lang: str = "ch"
    dpi: int = Field(default=144, ge=72, le=300)
    confidence_threshold: float = Field(default=DEFAULT_DETECT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    ocr_model_root: str | None = "model"
    use_doc_orientation: bool = False
    use_textline_orientation: bool = False
    use_doc_unwarping: bool = False
    det_thresh: float | None = None
    det_box_thresh: float | None = None
    drop_score: float | None = None
    return_word_box: bool = False
    ocr_batch_size: int = Field(default=DEFAULT_OCR_BATCH_SIZE, ge=1, le=32)


class OcrBoxResponse(BaseModel):
    id: str
    source: str
    bbox: list[float]
    polygon: list[list[float]] | None = None
    text: str | None = None
    confidence: float


class DetectPageResponse(BaseModel):
    page: int
    image_url: str
    width: int
    height: int
    boxes: list[OcrBoxResponse]


class DetectResponse(BaseModel):
    job_id: str
    status: str
    pages: list[DetectPageResponse]


class ApprovedBoxRequest(BaseModel):
    id: str
    source: str = "ocr-user"
    bbox: list[float]
    polygon: list[list[float]] | None = None
    text: str | None = None
    confidence: float = 1.0


class ApprovedBoxesPageRequest(BaseModel):
    page: int
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    boxes: list[ApprovedBoxRequest]


class ApprovedBoxesRequest(BaseModel):
    pages: list[ApprovedBoxesPageRequest]


class ApprovedBoxesResponse(BaseModel):
    job_id: str
    status: str
    approved_boxes_path: str
    pages: list[ApprovedBoxesPageRequest]


class ConvertRequest(BaseModel):
    mode: str = "editable"
    lang: str = "ch"
    ocr_model_root: str | None = "model"
    use_doc_orientation: bool = False
    use_textline_orientation: bool = False
    use_doc_unwarping: bool = False
    det_thresh: float | None = None
    det_box_thresh: float | None = None
    drop_score: float | None = None
    return_word_box: bool = False
    ocr_batch_size: int = Field(default=DEFAULT_OCR_BATCH_SIZE, ge=1, le=32)
    # None → server uses PDF2PPT_PAGE_TIMEOUT_S / default 120s.
    page_timeout_s: float | None = Field(default=None, ge=0.0, le=3600.0)
    dpi: int = Field(default=144, ge=72, le=300)
    background_dpi: int = Field(default=110, ge=72, le=300)
    background_format: str = "jpeg"
    background_jpeg_quality: int = Field(default=82, ge=1, le=95)
    inpaint_engine: Literal[
        "auto",
        "white-box",
        "opencv-fast",
        "lama-onnx",
        "lama-onnx-cuda",
        "lama-pytorch",
        "lama-onnx-hybrid",
        "lama-onnx-cuda-hybrid",
        "lama-pytorch-hybrid",
        "generative-openai",
        "generative-gemini",
    ] = DEFAULT_INPAINT_ENGINE
    inpaint_padding_px: int = Field(default=6, ge=0, le=128)
    inpaint_mask_dilate_px: int = Field(default=2, ge=0, le=32)
    inpaint_mask_dilate_line_height_ratio: float = Field(default=0.18, ge=0.0, le=1.0)
    inpaint_mask_dilate_min_px: int = Field(default=2, ge=0, le=64)
    inpaint_mask_dilate_max_px: int = Field(default=16, ge=0, le=64)
    inpaint_max_area_ratio: float = Field(default=0.12, ge=0.0, le=1.0)
    inpaint_model_root: str | None = "model/lama"
    inpaint_onnx_cuda_provider: str = "CUDAExecutionProvider"
    inpaint_onnx_execution_mode: Literal["sequential", "parallel"] = "sequential"
    inpaint_max_side_px: int = Field(default=1536, ge=256, le=8192)
    inpaint_lama_repo_root: str | None = "lama"
    inpaint_lama_device: str = "cuda"
    generative_model: str | None = None
    generative_max_calls: int = Field(default=50, ge=0, le=10_000)
    generative_max_cost_usd: float = Field(default=5.0, ge=0.0, le=10_000.0)
    generative_cache_dir: str | None = ".pdf2ppt_cache/generative_inpaint"
    generative_timeout_s: float = Field(default=120.0, ge=1.0, le=600.0)
    generative_verify_ocr: bool = True
    generative_low_freq_mae_threshold: float = Field(default=4.0, ge=-1.0, le=255.0)
    generative_skip_flat_ring: bool = True
    generative_ring_texture_measure: str = "highpass"
    generative_ring_texture_blur_sigma: float = Field(default=5.0, ge=0.0, le=64.0)
    generative_flat_ring_luma_std: float = Field(default=2.0, ge=0.0, le=255.0)
    generative_textured_ring_luma_std: float = Field(default=4.0, ge=0.0, le=255.0)
    generative_safety_block_retries: int = Field(default=1, ge=0, le=5)
    generative_max_attempts: int | None = Field(default=None, ge=1, le=50_000)
    vector_mode: Literal["off", "auto", "aggressive"] = "auto"
    vector_complexity_threshold: int = Field(default=80, ge=1, le=10_000)
    layout_engine: Literal["heuristic", "ppstructure"] = "heuristic"
    layout_model_root: str | None = "model"
    layout_skip_header_footer: bool = True
    # Experimental Phase-3 VLM layout planner (opt-in; never default/auto).
    # Model id, cache directory, and hard cost caps are server-side only
    # (env / process defaults) — clients may only enable the planner and
    # request low risk vs grouping/fonts.
    layout_planner: Literal["gemini"] | None = None
    layout_planner_max_calls: int = Field(default=30, ge=0, le=100)
    layout_planner_timeout_s: float = Field(default=45.0, ge=1.0, le=120.0)
    layout_planner_apply_grouping: bool = False
    layout_planner_apply_fonts: bool = False
    write_debug_artifacts: bool = False
    # P5 SlideIR — opt-in; defaults preserve convert behaviour.
    emit_ir: bool = False
    ir_only: bool = False
    # P4 render-compare-refine — opt-in / experimental; default off.
    refine: bool = False
    refine_iters: int = Field(default=2, ge=1, le=8)
    refine_max_fixes_per_slide: int = Field(default=8, ge=1, le=32)
    refine_max_attempts_per_slide: int = Field(default=8, ge=1, le=64)
    refine_time_budget_s: float = Field(default=90.0, ge=1.0, le=600.0)
    refine_min_font_size_pt: float = Field(default=8.0, ge=1.0, le=72.0)
    refine_min_font_ratio: float = Field(default=0.95, ge=0.1, le=1.0)


class ConvertResponse(BaseModel):
    job_id: str
    status: str
    output_pptx_path: str
    report_path: str
    page_count: int
    slideir_path: str | None = None
    slideir_assets_dir: str | None = None


class EmitIrResponse(BaseModel):
    job_id: str
    status: str
    slideir_path: str
    slideir_assets_dir: str
    report_path: str
    page_count: int


# --- Preview-before-convert (SlideIR overlays + light edits) ---------------

PreviewBlockRole = Literal["title", "body", "header", "footer", "page_number"]


class PreviewPageEdits(BaseModel):
    page: int = Field(ge=1)
    excluded_ids: list[str] = Field(default_factory=list, max_length=20_000)
    role_overrides: dict[str, PreviewBlockRole] = Field(default_factory=dict, max_length=20_000)
    raster_fallback: bool = False


class PreviewEditsRequest(BaseModel):
    pages: list[PreviewPageEdits] = Field(default_factory=list, max_length=10_000)


class PreviewEditsResponse(BaseModel):
    job_id: str
    status: str
    edits: PreviewEditsRequest


class PreviewTextBlock(BaseModel):
    id: str
    kind: Literal["text"] = "text"
    bbox: list[float]
    text: str = ""
    block_role: str = "body"
    reading_order: int = 0
    excluded: bool = False


class PreviewTableCell(BaseModel):
    row: int
    col: int
    text: str = ""
    bbox: list[float]


class PreviewTable(BaseModel):
    id: str
    kind: Literal["table"] = "table"
    bbox: list[float]
    row_count: int = 0
    col_count: int = 0
    cells: list[PreviewTableCell] = Field(default_factory=list)
    excluded: bool = False


class PreviewImage(BaseModel):
    id: str
    kind: Literal["image"] = "image"
    bbox: list[float]
    excluded: bool = False


class PreviewShape(BaseModel):
    id: str
    kind: Literal["shape"] = "shape"
    shape_kind: str = "rect"
    bbox: list[float]
    excluded: bool = False


class PreviewPageResponse(BaseModel):
    page: int
    image_url: str
    thumbnail_url: str | None = None
    width_pt: float
    height_pt: float
    page_kind: str = "digital"
    background_mode: str = "elements"
    baked_background: bool = False
    raster_fallback: bool = False
    text_blocks: list[PreviewTextBlock] = Field(default_factory=list)
    tables: list[PreviewTable] = Field(default_factory=list)
    images: list[PreviewImage] = Field(default_factory=list)
    shapes: list[PreviewShape] = Field(default_factory=list)


class PreviewResponse(BaseModel):
    job_id: str
    status: str
    page_count: int
    slide_width_pt: float
    slide_height_pt: float
    pages: list[PreviewPageResponse]
    edits: PreviewEditsRequest
    slideir_path: str | None = None
    slideir_assets_dir: str | None = None


class FromIrJobResponse(BaseModel):
    job_id: str
    status: str
    output_pptx_path: str
    report_path: str
    page_count: int
    slideir_path: str | None = None
    edited_slideir_path: str | None = None

