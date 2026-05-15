from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import fitz
import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from .api_models import (
    ApprovedBoxesRequest,
    ApprovedBoxesResponse,
    ConvertRequest,
    ConvertResponse,
    DetectPageResponse,
    DetectRequest,
    DetectResponse,
    ErrorResponse,
    JobResponse,
    OcrBoxResponse,
)
from .core import ConversionOptions, DEFAULT_OCR_BATCH_SIZE
from .core import InputValidationError, OcrInitializationError, OcrProcessingError, PageConversionError, Pdf2PptError
from .job_store import JobRecord, JobStore
from .models import TextBlock
from .ocr import OcrEngine
from .pipeline import convert_pdf

app = FastAPI(title="pdf2ppt API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
job_store = JobStore()


COMMON_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Input validation error."},
    404: {"model": ErrorResponse, "description": "Requested job or artifact was not found."},
    500: {"model": ErrorResponse, "description": "Page conversion or internal processing failure."},
    502: {"model": ErrorResponse, "description": "OCR runtime failed while processing a request."},
    503: {"model": ErrorResponse, "description": "OCR runtime is unavailable or misconfigured."},
}


@app.post("/jobs", response_model=JobResponse, responses={400: COMMON_ERROR_RESPONSES[400]})
async def create_job(file: UploadFile = File(...)) -> JobResponse:
    if not file.filename:
        raise _http_exception(400, "input-error", "Uploaded file must have a filename.")
    if not file.filename.lower().endswith(".pdf"):
        raise _http_exception(400, "input-error", "Only PDF uploads are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise _http_exception(400, "input-error", "Uploaded PDF is empty.")

    try:
        record = job_store.create_job(filename=file.filename, pdf_bytes=pdf_bytes)
    except Exception as error:
        raise _to_http_exception(InputValidationError(f"Failed to read uploaded PDF: {error}")) from error
    return _job_to_response(record)


@app.get("/jobs/{job_id}", response_model=JobResponse, responses={404: COMMON_ERROR_RESPONSES[404]})
def get_job(job_id: str) -> JobResponse:
    record = _load_job_or_404(job_id)
    return _job_to_response(record)


@app.delete("/jobs/{job_id}", status_code=204, responses={404: COMMON_ERROR_RESPONSES[404]})
def delete_job(job_id: str) -> Response:
    _load_job_or_404(job_id)
    job_store.delete_job(job_id)
    return Response(status_code=204)


@app.post(
    "/jobs/{job_id}/detect",
    response_model=DetectResponse,
    responses={
        404: COMMON_ERROR_RESPONSES[404],
        502: COMMON_ERROR_RESPONSES[502],
        503: COMMON_ERROR_RESPONSES[503],
    },
)
def detect_job(job_id: str, request: DetectRequest) -> DetectResponse:
    record = _load_job_or_404(job_id)
    job_store.update_job(job_id, status="detecting")

    ocr_engine = OcrEngine(
        lang=request.lang,
        model_root=Path(request.ocr_model_root) if request.ocr_model_root is not None else None,
        use_doc_orientation=request.use_doc_orientation,
        use_textline_orientation=request.use_textline_orientation,
        use_doc_unwarping=request.use_doc_unwarping,
        det_thresh=request.det_thresh,
        det_box_thresh=request.det_box_thresh,
        drop_score=request.drop_score,
    )

    pages: list[DetectPageResponse] = []
    try:
        with fitz.open(record.input_pdf_path) as document:
            for batch_start in range(0, document.page_count, request.ocr_batch_size):
                batch_pages = []
                batch_images = []
                batch_numbers = []
                for page_index in range(batch_start, min(batch_start + request.ocr_batch_size, document.page_count)):
                    page = document[page_index]
                    preview_image = _render_page_preview(page, dpi=request.dpi)
                    batch_pages.append(page_index)
                    batch_images.append(preview_image)
                    batch_numbers.append(page_index + 1)

                batch_ocr_results = ocr_engine.extract_text_blocks_batch(batch_images, batch_numbers)
                for page_index, preview_image, ocr_page_data in zip(batch_pages, batch_images, batch_ocr_results):
                    boxes = [
                        _text_block_to_box_response(block)
                        for block in ocr_page_data.blocks
                        if float(block.confidence) >= request.confidence_threshold
                    ]
                    pages.append(
                        DetectPageResponse(
                            page=page_index + 1,
                            image_url=f"/jobs/{job_id}/pages/{page_index + 1}.jpg",
                            width=preview_image.width,
                            height=preview_image.height,
                            boxes=boxes,
                        )
                    )
    except Exception as error:
        job_store.update_job(job_id, status="detect-failed")
        mapped_error = error if isinstance(error, (Pdf2PptError, OcrInitializationError, OcrProcessingError)) else OcrProcessingError(f"OCR detection failed: {error}")
        raise _to_http_exception(mapped_error) from error

    response = DetectResponse(job_id=job_id, status="detected", pages=pages)
    job_store.save_detection_payload(job_id, response.model_dump())
    return response


@app.put(
    "/jobs/{job_id}/boxes",
    response_model=ApprovedBoxesResponse,
    responses={400: COMMON_ERROR_RESPONSES[400], 404: COMMON_ERROR_RESPONSES[404]},
)
def save_approved_boxes(job_id: str, request: ApprovedBoxesRequest) -> ApprovedBoxesResponse:
    _load_job_or_404(job_id)
    payload = {"job_id": job_id, "status": "boxes-approved", "pages": request.model_dump()["pages"]}
    output_path = job_store.save_approved_boxes_payload(job_id, payload)
    return ApprovedBoxesResponse(
        job_id=job_id,
        status="boxes-approved",
        approved_boxes_path=str(output_path),
        pages=request.pages,
    )


@app.post(
    "/jobs/{job_id}/convert",
    response_model=ConvertResponse,
    responses={
        400: COMMON_ERROR_RESPONSES[400],
        404: COMMON_ERROR_RESPONSES[404],
        500: COMMON_ERROR_RESPONSES[500],
        502: COMMON_ERROR_RESPONSES[502],
        503: COMMON_ERROR_RESPONSES[503],
    },
)
def convert_job(job_id: str, request: ConvertRequest) -> ConvertResponse:
    record = _load_job_or_404(job_id)
    if record.approved_boxes_path is None:
        raise _to_http_exception(InputValidationError("No approved boxes found for this job."))

    approved_boxes_by_page, approved_image_sizes_by_page = _load_approved_boxes_payload(Path(record.approved_boxes_path))
    output_pptx_path = job_store.job_dir(job_id) / "output.pptx"
    report_path = job_store.job_dir(job_id) / "output.report.json"
    debug_dir = job_store.job_dir(job_id) / "debug" if request.write_debug_artifacts else None

    options = ConversionOptions(
        input_path=Path(record.input_pdf_path),
        output_path=output_pptx_path,
        report_path=report_path,
        mode=request.mode,
        lang=request.lang,
        ocr_model_root=Path(request.ocr_model_root) if request.ocr_model_root is not None else None,
        ocr_use_doc_orientation=request.use_doc_orientation,
        ocr_use_textline_orientation=request.use_textline_orientation,
        ocr_det_thresh=request.det_thresh,
        ocr_det_box_thresh=request.det_box_thresh,
        ocr_drop_score=request.drop_score,
        ocr_batch_size=request.ocr_batch_size,
        render_dpi=request.dpi,
        background_dpi=request.background_dpi,
        background_image_format=request.background_format,
        background_jpeg_quality=request.background_jpeg_quality,
        debug_dir=debug_dir,
        use_doc_unwarping=request.use_doc_unwarping,
        inpaint_engine=request.inpaint_engine,
        inpaint_padding_px=request.inpaint_padding_px,
        inpaint_max_area_ratio=request.inpaint_max_area_ratio,
        inpaint_model_root=Path(request.inpaint_model_root) if request.inpaint_model_root is not None else None,
        inpaint_onnx_cuda_provider=request.inpaint_onnx_cuda_provider,
        inpaint_onnx_execution_mode=request.inpaint_onnx_execution_mode,
        inpaint_max_side_px=request.inpaint_max_side_px,
        approved_ocr_blocks_by_page=approved_boxes_by_page,
        approved_ocr_image_size_by_page=approved_image_sizes_by_page,
    )

    job_store.update_job(job_id, status="converting")
    try:
        report = convert_pdf(options)
    except (InputValidationError, OcrInitializationError, OcrProcessingError, PageConversionError) as error:
        job_store.update_job(job_id, status="convert-failed")
        raise _to_http_exception(error) from error
    except Exception as error:
        job_store.update_job(job_id, status="convert-failed")
        raise _to_http_exception(PageConversionError(0, f"Conversion failed: {error}")) from error

    job_store.update_job(
        job_id,
        status="converted",
        output_pptx_path=str(output_pptx_path),
        report_path=str(report_path),
    )
    return ConvertResponse(
        job_id=job_id,
        status="converted",
        output_pptx_path=str(output_pptx_path),
        report_path=str(report_path),
        page_count=len(report.pages),
    )


@app.get("/jobs/{job_id}/pages/{page_number}.jpg", responses={404: COMMON_ERROR_RESPONSES[404]})
def get_job_page_preview(job_id: str, page_number: int) -> FileResponse:
    record = _load_job_or_404(job_id)
    try:
        with fitz.open(record.input_pdf_path) as document:
            if page_number < 1 or page_number > document.page_count:
                raise _http_exception(404, "not-found", "Preview page not found.")
            preview_image = _render_page_preview(document[page_number - 1], dpi=144)
    except HTTPException:
        raise
    except Exception as error:
        raise _to_http_exception(InputValidationError(f"Failed to render preview image: {error}")) from error

    buffer = io.BytesIO()
    preview_image.save(buffer, format="JPEG", quality=85, optimize=True)
    buffer.seek(0)
    return Response(content=buffer.getvalue(), media_type="image/jpeg")


@app.get("/jobs/{job_id}/output.pptx", responses={404: COMMON_ERROR_RESPONSES[404]})
def download_output_pptx(job_id: str) -> FileResponse:
    record = _load_job_or_404(job_id)
    if record.output_pptx_path is None:
        raise _http_exception(404, "not-found", "Converted PPTX not found.")
    output_path = Path(record.output_pptx_path)
    if not output_path.exists():
        raise _http_exception(404, "not-found", "Converted PPTX file is missing.")
    download_filename = f"{Path(record.original_filename).stem}.pptx"
    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=download_filename,
    )


@app.get("/jobs/{job_id}/report.json", responses={404: COMMON_ERROR_RESPONSES[404]})
def download_report_json(job_id: str) -> FileResponse:
    record = _load_job_or_404(job_id)
    if record.report_path is None:
        raise _http_exception(404, "not-found", "Conversion report not found.")
    report_path = Path(record.report_path)
    if not report_path.exists():
        raise _http_exception(404, "not-found", "Conversion report file is missing.")
    return FileResponse(report_path, media_type="application/json")


def run() -> None:
    uvicorn.run("pdf2ppt.api:app", host="0.0.0.0", port=8008, reload=False)


def _load_job_or_404(job_id: str) -> JobRecord:
    try:
        return job_store.get_job(job_id)
    except FileNotFoundError as error:
        raise _http_exception(404, "not-found", "Job not found.") from error


def _job_to_response(record: JobRecord) -> JobResponse:
    return JobResponse(
        job_id=record.job_id,
        status=record.status,
        original_filename=record.original_filename,
        page_count=record.page_count,
        created_at=record.created_at,
        updated_at=record.updated_at,
        detection_path=record.detection_path,
        approved_boxes_path=record.approved_boxes_path,
        output_pptx_path=record.output_pptx_path,
        report_path=record.report_path,
    )


def _render_page_preview(page: fitz.Page, *, dpi: int) -> Image.Image:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


def _text_block_to_box_response(block: Any) -> OcrBoxResponse:
    polygon = None
    if block.image_polygon is not None:
        polygon = [[float(point[0]), float(point[1])] for point in block.image_polygon]
    return OcrBoxResponse(
        id=block.id,
        source="ocr-auto",
        bbox=[float(value) for value in block.bbox],
        polygon=polygon,
        text=block.text,
        confidence=float(block.confidence),
    )


def _load_approved_boxes_payload(
    approved_boxes_path: Path,
) -> tuple[dict[int, list[TextBlock]], dict[int, tuple[int, int]]]:
    payload = approved_boxes_path.read_text(encoding="utf-8")
    data = json.loads(payload)
    blocks_by_page: dict[int, list[TextBlock]] = {}
    image_sizes_by_page: dict[int, tuple[int, int]] = {}
    for page_payload in data.get("pages", []):
        page_number = int(page_payload["page"])
        width = int(page_payload["width"])
        height = int(page_payload["height"])
        if width < 1 or height < 1:
            raise InputValidationError(f"Approved box image size must be positive for page {page_number}.")
        image_sizes_by_page[page_number] = (width, height)
        blocks: list[TextBlock] = []
        for index, box in enumerate(page_payload.get("boxes", []), start=1):
            polygon = box.get("polygon")
            bbox = box.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise InputValidationError(f"Approved box {index} on page {page_number} must contain a 4-value bbox.")
            blocks.append(
                TextBlock(
                    id=str(box.get("id") or f"approved_{page_number}_{index}"),
                    source="ocr",
                    bbox=tuple(float(value) for value in bbox),
                    text=str(box.get("text") or ""),
                    confidence=float(box.get("confidence", 1.0)),
                    image_bbox=tuple(float(value) for value in bbox),
                    image_polygon=(
                        tuple((float(point[0]), float(point[1])) for point in polygon)
                        if polygon is not None
                        else None
                    ),
                )
            )
        blocks_by_page[page_number] = blocks
    return blocks_by_page, image_sizes_by_page


def _to_http_exception(error: Exception) -> HTTPException:
    if isinstance(error, InputValidationError):
        return HTTPException(status_code=400, detail=error.to_detail())
    if isinstance(error, OcrInitializationError):
        return HTTPException(status_code=503, detail=error.to_detail())
    if isinstance(error, OcrProcessingError):
        return HTTPException(status_code=502, detail=error.to_detail())
    if isinstance(error, PageConversionError):
        return HTTPException(status_code=500, detail=error.to_detail())
    if isinstance(error, Pdf2PptError):
        return HTTPException(status_code=500, detail=error.to_detail())
    return _http_exception(500, "internal-error", str(error))


def _http_exception(status_code: int, code: str, message: str, page: int | None = None) -> HTTPException:
    detail: dict[str, object] = {"code": code, "message": message}
    if page is not None:
        detail["page"] = page
    return HTTPException(status_code=status_code, detail=detail)
