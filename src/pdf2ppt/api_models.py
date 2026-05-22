from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .core import DEFAULT_OCR_BATCH_SIZE


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
    dpi: int = Field(default=144, ge=72, le=300)
    background_dpi: int = Field(default=110, ge=72, le=300)
    background_format: str = "jpeg"
    background_jpeg_quality: int = Field(default=82, ge=1, le=95)
    inpaint_engine: Literal[
        "auto",
        "white-box",
        "opencv-fast",
        "lama-onnx-cuda",
        "lama-pytorch",
        "lama-onnx-cuda-hybrid",
        "lama-pytorch-hybrid",
    ] = "opencv-fast"
    inpaint_padding_px: int = Field(default=6, ge=0, le=128)
    inpaint_max_area_ratio: float = Field(default=0.12, ge=0.0, le=1.0)
    inpaint_model_root: str | None = "model/lama"
    inpaint_onnx_cuda_provider: str = "CUDAExecutionProvider"
    inpaint_onnx_execution_mode: Literal["sequential", "parallel"] = "sequential"
    inpaint_max_side_px: int = Field(default=1536, ge=256, le=8192)
    inpaint_lama_repo_root: str | None = "lama"
    inpaint_lama_device: str = "cuda"
    write_debug_artifacts: bool = False


class ConvertResponse(BaseModel):
    job_id: str
    status: str
    output_pptx_path: str
    report_path: str
    page_count: int
