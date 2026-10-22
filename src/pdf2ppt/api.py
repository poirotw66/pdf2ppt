from __future__ import annotations

import io
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import fitz
import uvicorn
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from starlette.background import BackgroundTask

from .api_models import (
    ApprovedBoxesRequest,
    ApprovedBoxesResponse,
    ConvertRequest,
    ConvertResponse,
    DetectPageResponse,
    DetectRequest,
    DetectResponse,
    EmitIrResponse,
    ErrorResponse,
    JobResponse,
    OcrBoxResponse,
)
from .background import OpenCvFastInpaintingEngine, build_text_mask_image
from .core import (
    ConversionOptions,
    InputValidationError,
    OcrInitializationError,
    OcrProcessingError,
    PageConversionError,
    Pdf2PptError,
)
from .inpainting_engines import base_lama_inpaint_engine, uses_lama_patch_hybrid
from .ir_io import SlideIrError, default_assets_dir_name
from .job_store import JobRecord, JobStore
from .models import TextBlock
from .ocr import OcrEngine
from .paths import resolve_repo_relative_path
from .pipeline import build_notebooklm_watermark_mask_block, convert_from_ir, convert_pdf
from .upload_input import SUPPORTED_UPLOAD_SUFFIXES, is_raster_image_upload, normalize_upload_to_pdf_bytes

logger = logging.getLogger(__name__)

# Caps for POST /from-ir uploads (agent-facing; keep modest for local job store).
MAX_FROM_IR_FILE_BYTES = 50 * 1024 * 1024
MAX_FROM_IR_ZIP_MEMBERS = 10_000

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
    if Path(file.filename).suffix.lower() not in SUPPORTED_UPLOAD_SUFFIXES:
        raise _http_exception(400, "input-error", "Only PDF, PNG, and JPG uploads are supported.")

    file_bytes = await file.read()
    try:
        pdf_bytes = normalize_upload_to_pdf_bytes(filename=file.filename, file_bytes=file_bytes)
        record = job_store.create_job(filename=file.filename, pdf_bytes=pdf_bytes)
    except InputValidationError as error:
        raise _to_http_exception(error) from error
    except Exception as error:
        raise _to_http_exception(InputValidationError(f"Failed to process uploaded file: {error}")) from error
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
        model_root=resolve_repo_relative_path(request.ocr_model_root),
        use_doc_orientation=request.use_doc_orientation,
        use_textline_orientation=request.use_textline_orientation,
        use_doc_unwarping=request.use_doc_unwarping,
        det_thresh=request.det_thresh,
        det_box_thresh=request.det_box_thresh,
        drop_score=request.drop_score,
        return_word_box=request.return_word_box,
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
                    preview_image = _render_page_preview(
                        page,
                        dpi=request.dpi,
                        apply_notebooklm_fallback=_should_apply_notebooklm_watermark_fallback(record.original_filename),
                    )
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
    if request.ir_only and not request.emit_ir:
        raise _to_http_exception(InputValidationError("ir_only requires emit_ir=true."))

    approved_boxes_by_page, approved_image_sizes_by_page = _load_approved_boxes_payload(Path(record.approved_boxes_path))
    job_dir = job_store.job_dir(job_id)
    output_pptx_path = job_dir / "output.pptx"
    report_path = job_dir / "output.report.json"
    debug_dir = job_dir / "debug" if request.write_debug_artifacts else None
    slideir_path = job_dir / "output.slideir.json" if request.emit_ir else None
    write_pptx = not request.ir_only

    options = _conversion_options_from_request(
        request,
        input_path=Path(record.input_pdf_path),
        output_path=output_pptx_path,
        report_path=report_path,
        debug_dir=debug_dir,
        approved_ocr_blocks_by_page=approved_boxes_by_page,
        approved_ocr_image_size_by_page=approved_image_sizes_by_page,
        apply_notebooklm_watermark_fallback=_should_apply_notebooklm_watermark_fallback(record.original_filename),
    )

    job_store.update_job(job_id, status="converting")
    try:
        report = convert_pdf(
            options,
            write_pptx=write_pptx,
            emit_ir_path=slideir_path,
        )
    except (InputValidationError, OcrInitializationError, OcrProcessingError, PageConversionError) as error:
        job_store.update_job(job_id, status="convert-failed")
        raise _to_http_exception(error) from error
    except Exception as error:
        job_store.update_job(job_id, status="convert-failed")
        raise _to_http_exception(PageConversionError(0, f"Conversion failed: {error}")) from error

    assets_dir = None
    if slideir_path is not None:
        assets_dir = str(job_dir / default_assets_dir_name(slideir_path))

    # Preserve any prior PPTX path on ir_only converts (do not clear the field).
    update_fields: dict[str, object] = {
        "status": "converted",
        "report_path": str(report_path),
    }
    if write_pptx:
        update_fields["output_pptx_path"] = str(output_pptx_path)
    job_store.update_job(job_id, **update_fields)
    return ConvertResponse(
        job_id=job_id,
        status="converted",
        output_pptx_path=str(output_pptx_path) if write_pptx else (record.output_pptx_path or ""),
        report_path=str(report_path),
        page_count=len(report.pages),
        slideir_path=str(slideir_path) if slideir_path is not None else None,
        slideir_assets_dir=assets_dir,
    )


@app.post(
    "/jobs/{job_id}/emit-ir",
    response_model=EmitIrResponse,
    responses={
        400: COMMON_ERROR_RESPONSES[400],
        404: COMMON_ERROR_RESPONSES[404],
        500: COMMON_ERROR_RESPONSES[500],
        502: COMMON_ERROR_RESPONSES[502],
        503: COMMON_ERROR_RESPONSES[503],
    },
)
def emit_ir_job(job_id: str, request: ConvertRequest) -> EmitIrResponse:
    """Analyze the job PDF and write SlideIR only (no PPTX). Opt-in agent path."""
    record = _load_job_or_404(job_id)
    # Approved boxes are optional for emit-ir: agents may want raw analysis.
    approved_boxes_by_page: dict[int, list[TextBlock]] = {}
    approved_image_sizes_by_page: dict[int, tuple[int, int]] = {}
    if record.approved_boxes_path is not None:
        approved_boxes_by_page, approved_image_sizes_by_page = _load_approved_boxes_payload(
            Path(record.approved_boxes_path)
        )

    job_dir = job_store.job_dir(job_id)
    slideir_path = job_dir / "output.slideir.json"
    report_path = job_dir / "output.slideir.report.json"
    placeholder_pptx = job_dir / "output.pptx"
    debug_dir = job_dir / "debug" if request.write_debug_artifacts else None

    options = _conversion_options_from_request(
        request,
        input_path=Path(record.input_pdf_path),
        output_path=placeholder_pptx,
        report_path=report_path,
        debug_dir=debug_dir,
        approved_ocr_blocks_by_page=approved_boxes_by_page,
        approved_ocr_image_size_by_page=approved_image_sizes_by_page,
        apply_notebooklm_watermark_fallback=_should_apply_notebooklm_watermark_fallback(record.original_filename),
    )

    job_store.update_job(job_id, status="emitting-ir")
    try:
        report = convert_pdf(options, write_pptx=False, emit_ir_path=slideir_path)
    except (InputValidationError, OcrInitializationError, OcrProcessingError, PageConversionError) as error:
        job_store.update_job(job_id, status="emit-ir-failed")
        raise _to_http_exception(error) from error
    except Exception as error:
        job_store.update_job(job_id, status="emit-ir-failed")
        raise _to_http_exception(PageConversionError(0, f"Emit-IR failed: {error}")) from error

    assets_dir = job_dir / default_assets_dir_name(slideir_path)
    job_store.update_job(job_id, status="ir-emitted", report_path=str(report_path))
    return EmitIrResponse(
        job_id=job_id,
        status="ir-emitted",
        slideir_path=str(slideir_path),
        slideir_assets_dir=str(assets_dir),
        report_path=str(report_path),
        page_count=len(report.pages),
    )


@app.post(
    "/from-ir",
    responses={400: COMMON_ERROR_RESPONSES[400], 500: COMMON_ERROR_RESPONSES[500]},
)
async def from_ir_upload(
    ir_file: UploadFile = File(..., description="SlideIR JSON file"),
    assets_zip: UploadFile | None = File(
        default=None,
        description="Optional zip of the assets_dir (paths relative to assets root).",
    ),
) -> FileResponse:
    """Render an uploaded SlideIR JSON (+ optional assets zip) and return the PPTX."""
    if not ir_file.filename:
        raise _http_exception(400, "input-error", "SlideIR upload must have a filename.")

    work = Path(tempfile.mkdtemp(prefix="pdf2ppt-from-ir-", dir=str(job_store.root)))
    try:
        ir_path = work / Path(ir_file.filename).name
        ir_bytes = await ir_file.read()
        _reject_oversized_upload(len(ir_bytes), label="SlideIR JSON")
        ir_path.write_bytes(ir_bytes)

        if assets_zip is not None and assets_zip.filename:
            zip_bytes = await assets_zip.read()
            _reject_oversized_upload(len(zip_bytes), label="assets zip")
            zip_path = work / "assets.zip"
            zip_path.write_bytes(zip_bytes)
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    _safe_extract_zip(archive, work)
            except zipfile.BadZipFile as error:
                raise _to_http_exception(InputValidationError(f"Invalid assets zip: {error}")) from error
            except InputValidationError as error:
                raise _to_http_exception(error) from error

            try:
                payload = json.loads(ir_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise _to_http_exception(InputValidationError(f"Invalid SlideIR JSON: {error}")) from error
            assets_name = str(payload.get("assets_dir") or default_assets_dir_name(ir_path))
            assets_dir = work / assets_name
            if not assets_dir.exists():
                nested = work / "pages"
                if nested.exists():
                    assets_dir.mkdir(parents=True, exist_ok=True)
                    nested.rename(assets_dir / "pages")

        output_pptx = work / "output.pptx"
        report_path = work / "output.report.json"
        try:
            convert_from_ir(ir_path, output_pptx, report_path=report_path)
        except SlideIrError as error:
            raise _to_http_exception(_scrub_slideir_error(error)) from error
        except Exception as error:
            raise _to_http_exception(PageConversionError(0, "from-ir failed during render.")) from error

        download_name = f"{Path(ir_file.filename).stem}.pptx"
        return FileResponse(
            output_pptx,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=download_name,
            background=BackgroundTask(shutil.rmtree, work, ignore_errors=True),
        )
    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except (InputValidationError, SlideIrError) as error:
        shutil.rmtree(work, ignore_errors=True)
        raise _to_http_exception(error) from error
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


@app.get("/jobs/{job_id}/slideir.json", responses={404: COMMON_ERROR_RESPONSES[404]})
def download_slideir_json(job_id: str) -> FileResponse:
    record = _load_job_or_404(job_id)
    slideir_path = _resolve_job_slideir_path(record)
    if slideir_path is None or not slideir_path.exists():
        raise _http_exception(404, "not-found", "SlideIR JSON not found for this job.")
    return FileResponse(
        slideir_path,
        media_type="application/json",
        filename=f"{Path(record.original_filename).stem}.slideir.json",
    )


@app.get("/jobs/{job_id}/slideir-assets.zip", responses={404: COMMON_ERROR_RESPONSES[404]})
def download_slideir_assets_zip(job_id: str) -> FileResponse:
    record = _load_job_or_404(job_id)
    slideir_path = _resolve_job_slideir_path(record)
    if slideir_path is None or not slideir_path.exists():
        raise _http_exception(404, "not-found", "SlideIR JSON not found for this job.")
    assets_dir = job_store.job_dir(job_id) / default_assets_dir_name(slideir_path)
    if not assets_dir.is_dir():
        raise _http_exception(404, "not-found", "SlideIR assets directory not found for this job.")
    zip_path = job_store.job_dir(job_id) / "slideir-assets.zip"
    _zip_directory(assets_dir, zip_path)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{Path(record.original_filename).stem}.slideir.assets.zip",
    )


@app.get("/jobs/{job_id}/pages/{page_number}.jpg", responses={404: COMMON_ERROR_RESPONSES[404]})
def get_job_page_preview(job_id: str, page_number: int) -> Response:
    record = _load_job_or_404(job_id)
    try:
        with fitz.open(record.input_pdf_path) as document:
            if page_number < 1 or page_number > document.page_count:
                raise _http_exception(404, "not-found", "Preview page not found.")
            preview_image = _render_page_preview(
                document[page_number - 1],
                dpi=144,
                apply_notebooklm_fallback=_should_apply_notebooklm_watermark_fallback(record.original_filename),
            )
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


def _reject_oversized_upload(size: int, *, label: str) -> None:
    if size > MAX_FROM_IR_FILE_BYTES:
        raise _to_http_exception(
            InputValidationError(
                f"{label} exceeds the {MAX_FROM_IR_FILE_BYTES // (1024 * 1024)} MiB upload limit."
            )
        )


def _safe_extract_zip(archive: zipfile.ZipFile, dest: Path) -> None:
    """Extract a zip into ``dest``, rejecting path-traversal members."""
    dest_resolved = dest.resolve()
    members = archive.infolist()
    if len(members) > MAX_FROM_IR_ZIP_MEMBERS:
        raise InputValidationError(
            f"assets zip has too many members (limit {MAX_FROM_IR_ZIP_MEMBERS})."
        )
    for info in members:
        name = info.filename
        if not name or name.endswith("/"):
            continue
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise InputValidationError("Unsafe path in assets zip.")
        target = (dest / path).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as error:
            raise InputValidationError("Unsafe path in assets zip.") from error
    archive.extractall(dest)


def _scrub_slideir_error(error: SlideIrError) -> SlideIrError:
    """Rewrite SlideIR errors so absolute local paths are not returned to clients."""
    message = str(error)
    scrubbed = message
    # Drop common absolute-path prefixes that leak host layout.
    for marker in ("Missing SlideIR asset: ", "Path escapes assets root: ", "SlideIR file does not exist: "):
        if marker in scrubbed:
            scrubbed = marker + Path(scrubbed.split(marker, 1)[1]).name
            break
    if scrubbed == message:
        return error
    return SlideIrError(scrubbed)


def _resolve_job_slideir_path(record: JobRecord) -> Path | None:
    job_dir = job_store.job_dir(record.job_id)
    candidates = (
        job_dir / "output.slideir.json",
        Path(record.report_path).with_name("output.slideir.json") if record.report_path else None,
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(source_dir)))


def _conversion_options_from_request(
    request: ConvertRequest,
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    debug_dir: Path | None,
    approved_ocr_blocks_by_page: dict[int, list[TextBlock]],
    approved_ocr_image_size_by_page: dict[int, tuple[int, int]],
    apply_notebooklm_watermark_fallback: bool,
) -> ConversionOptions:
    return ConversionOptions(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        mode=request.mode,
        lang=request.lang,
        ocr_model_root=resolve_repo_relative_path(request.ocr_model_root),
        ocr_use_doc_orientation=request.use_doc_orientation,
        ocr_use_textline_orientation=request.use_textline_orientation,
        ocr_det_thresh=request.det_thresh,
        ocr_det_box_thresh=request.det_box_thresh,
        ocr_drop_score=request.drop_score,
        ocr_return_word_box=request.return_word_box,
        ocr_batch_size=request.ocr_batch_size,
        render_dpi=request.dpi,
        background_dpi=request.background_dpi,
        background_image_format=request.background_format,
        background_jpeg_quality=request.background_jpeg_quality,
        debug_dir=debug_dir,
        use_doc_unwarping=request.use_doc_unwarping,
        inpaint_engine=base_lama_inpaint_engine(request.inpaint_engine),
        inpaint_lama_patch_hybrid=uses_lama_patch_hybrid(request.inpaint_engine),
        inpaint_padding_px=request.inpaint_padding_px,
        inpaint_mask_dilate_px=request.inpaint_mask_dilate_px,
        inpaint_mask_dilate_line_height_ratio=request.inpaint_mask_dilate_line_height_ratio,
        inpaint_mask_dilate_min_px=request.inpaint_mask_dilate_min_px,
        inpaint_mask_dilate_max_px=request.inpaint_mask_dilate_max_px,
        inpaint_max_area_ratio=request.inpaint_max_area_ratio,
        inpaint_model_root=resolve_repo_relative_path(request.inpaint_model_root),
        inpaint_onnx_cuda_provider=request.inpaint_onnx_cuda_provider,
        inpaint_onnx_execution_mode=request.inpaint_onnx_execution_mode,
        inpaint_max_side_px=request.inpaint_max_side_px,
        inpaint_lama_repo_root=resolve_repo_relative_path(request.inpaint_lama_repo_root),
        inpaint_lama_device=request.inpaint_lama_device,
        generative_model=request.generative_model,
        generative_max_calls=request.generative_max_calls,
        generative_max_cost_usd=request.generative_max_cost_usd,
        generative_cache_dir=(Path(request.generative_cache_dir) if request.generative_cache_dir else None),
        generative_timeout_s=request.generative_timeout_s,
        generative_verify_ocr=request.generative_verify_ocr,
        generative_low_freq_mae_threshold=request.generative_low_freq_mae_threshold,
        generative_skip_flat_ring=request.generative_skip_flat_ring,
        generative_ring_texture_measure=request.generative_ring_texture_measure,
        generative_ring_texture_blur_sigma=request.generative_ring_texture_blur_sigma,
        generative_flat_ring_luma_std=request.generative_flat_ring_luma_std,
        generative_textured_ring_luma_std=request.generative_textured_ring_luma_std,
        generative_safety_block_retries=request.generative_safety_block_retries,
        generative_max_attempts=request.generative_max_attempts,
        vector_mode=request.vector_mode,
        vector_complexity_threshold=request.vector_complexity_threshold,
        layout_engine=request.layout_engine,
        layout_model_root=Path(request.layout_model_root) if request.layout_model_root else None,
        layout_skip_header_footer=request.layout_skip_header_footer,
        layout_planner=request.layout_planner,
        layout_planner_model=request.layout_planner_model,
        layout_planner_max_calls=request.layout_planner_max_calls,
        layout_planner_max_cost_usd=request.layout_planner_max_cost_usd,
        layout_planner_timeout_s=request.layout_planner_timeout_s,
        layout_planner_concurrency=request.layout_planner_concurrency,
        layout_planner_cache_dir=(
            Path(request.layout_planner_cache_dir) if request.layout_planner_cache_dir else None
        ),
        layout_planner_min_confidence=request.layout_planner_min_confidence,
        approved_ocr_blocks_by_page=approved_ocr_blocks_by_page,
        approved_ocr_image_size_by_page=approved_ocr_image_size_by_page,
        apply_notebooklm_watermark_fallback=apply_notebooklm_watermark_fallback,
        refine=bool(getattr(request, "refine", False)),
        refine_iters=int(getattr(request, "refine_iters", 2)),
        refine_max_fixes_per_slide=int(getattr(request, "refine_max_fixes_per_slide", 8)),
        refine_max_attempts_per_slide=int(getattr(request, "refine_max_attempts_per_slide", 8)),
        refine_time_budget_s=float(getattr(request, "refine_time_budget_s", 90.0)),
        refine_min_font_size_pt=float(getattr(request, "refine_min_font_size_pt", 8.0)),
        refine_min_font_ratio=float(getattr(request, "refine_min_font_ratio", 0.95)),
    )


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


def _should_apply_notebooklm_watermark_fallback(original_filename: str) -> bool:
    return not is_raster_image_upload(original_filename)


def _render_page_preview(page: fitz.Page, *, dpi: int, apply_notebooklm_fallback: bool = True) -> Image.Image:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    preview_image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return _clean_preview_watermark(preview_image, page.rect, apply_notebooklm_fallback=apply_notebooklm_fallback)


def _clean_preview_watermark(
    preview_image: Image.Image,
    page_rect: fitz.Rect,
    *,
    apply_notebooklm_fallback: bool = True,
) -> Image.Image:
    if not apply_notebooklm_fallback:
        return preview_image
    watermark_block = build_notebooklm_watermark_mask_block(page_rect)
    mask_image = build_text_mask_image([watermark_block], preview_image.size, page_rect, padding_px=2)
    try:
        return OpenCvFastInpaintingEngine().inpaint(preview_image, mask_image)
    except Exception as error:
        logger.warning("Preview watermark cleanup failed; returning original preview: %s", error)
        return preview_image


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
            bbox_tuple = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            blocks.append(
                TextBlock(
                    id=str(box.get("id") or f"approved_{page_number}_{index}"),
                    source="ocr",
                    bbox=bbox_tuple,
                    text=str(box.get("text") or ""),
                    confidence=float(box.get("confidence", 1.0)),
                    image_bbox=bbox_tuple,
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
