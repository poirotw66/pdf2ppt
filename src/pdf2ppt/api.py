from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
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
    JobResponse,
    OcrBoxResponse,
)
from .core import ConversionOptions, DEFAULT_OCR_BATCH_SIZE
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


@app.post("/jobs", response_model=JobResponse)
async def create_job(file: UploadFile = File(...)) -> JobResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    try:
        record = job_store.create_job(filename=file.filename, pdf_bytes=pdf_bytes)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Failed to create job: {error}") from error
    return _job_to_response(record)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    record = _load_job_or_404(job_id)
    return _job_to_response(record)


@app.post("/jobs/{job_id}/detect", response_model=DetectResponse)
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
                    preview_path = job_store.preview_image_path(job_id, page_index + 1)
                    preview_image.save(preview_path, format="PNG")
                    batch_pages.append(page_index)
                    batch_images.append(preview_image)
                    batch_numbers.append(page_index + 1)

                batch_ocr_results = ocr_engine.extract_text_blocks_batch(batch_images, batch_numbers)
                for page_index, preview_image, ocr_page_data in zip(batch_pages, batch_images, batch_ocr_results):
                    boxes = [_text_block_to_box_response(block) for block in ocr_page_data.blocks]
                    pages.append(
                        DetectPageResponse(
                            page=page_index + 1,
                            image_url=f"/jobs/{job_id}/pages/{page_index + 1}.png",
                            width=preview_image.width,
                            height=preview_image.height,
                            boxes=boxes,
                        )
                    )
    except Exception as error:
        job_store.update_job(job_id, status="detect-failed")
        raise HTTPException(status_code=500, detail=f"OCR detection failed: {error}") from error

    response = DetectResponse(job_id=job_id, status="detected", pages=pages)
    job_store.save_detection_payload(job_id, response.model_dump())
    return response


@app.put("/jobs/{job_id}/boxes", response_model=ApprovedBoxesResponse)
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


@app.post("/jobs/{job_id}/convert", response_model=ConvertResponse)
def convert_job(job_id: str, request: ConvertRequest) -> ConvertResponse:
    record = _load_job_or_404(job_id)
    if record.approved_boxes_path is None:
        raise HTTPException(status_code=400, detail="No approved boxes found for this job.")

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
        approved_ocr_blocks_by_page=approved_boxes_by_page,
        approved_ocr_image_size_by_page=approved_image_sizes_by_page,
    )

    job_store.update_job(job_id, status="converting")
    try:
        report = convert_pdf(options)
    except Exception as error:
        job_store.update_job(job_id, status="convert-failed")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {error}") from error

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


@app.get("/jobs/{job_id}/pages/{page_number}.png")
def get_job_page_preview(job_id: str, page_number: int) -> FileResponse:
    _load_job_or_404(job_id)
    preview_path = job_store.preview_image_path(job_id, page_number)
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview image not found.")
    return FileResponse(preview_path, media_type="image/png")


@app.get("/jobs/{job_id}/output.pptx")
def download_output_pptx(job_id: str) -> FileResponse:
    record = _load_job_or_404(job_id)
    if record.output_pptx_path is None:
        raise HTTPException(status_code=404, detail="Converted PPTX not found.")
    output_path = Path(record.output_pptx_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Converted PPTX file is missing.")
    return FileResponse(output_path, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")


@app.get("/jobs/{job_id}/report.json")
def download_report_json(job_id: str) -> FileResponse:
    record = _load_job_or_404(job_id)
    if record.report_path is None:
        raise HTTPException(status_code=404, detail="Conversion report not found.")
    report_path = Path(record.report_path)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Conversion report file is missing.")
    return FileResponse(report_path, media_type="application/json")


def run() -> None:
    uvicorn.run("pdf2ppt.api:app", host="0.0.0.0", port=8000, reload=False)


def _load_job_or_404(job_id: str) -> JobRecord:
    try:
        return job_store.get_job(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found.") from error


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
        image_sizes_by_page[page_number] = (width, height)
        blocks: list[TextBlock] = []
        for index, box in enumerate(page_payload.get("boxes", []), start=1):
            polygon = box.get("polygon")
            blocks.append(
                TextBlock(
                    id=str(box.get("id") or f"approved_{page_number}_{index}"),
                    source="ocr",
                    bbox=tuple(float(value) for value in box["bbox"]),
                    text=str(box.get("text") or ""),
                    confidence=float(box.get("confidence", 1.0)),
                    image_bbox=tuple(float(value) for value in box["bbox"]),
                    image_polygon=(
                        tuple((float(point[0]), float(point[1])) for point in polygon)
                        if polygon is not None
                        else None
                    ),
                )
            )
        blocks_by_page[page_number] = blocks
    return blocks_by_page, image_sizes_by_page
