from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

from .core import InputValidationError

SUPPORTED_UPLOAD_SUFFIXES = frozenset({".pdf", ".png", ".jpg", ".jpeg"})
RASTER_IMAGE_UPLOAD_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def is_raster_image_upload(filename: str) -> bool:
    return Path(filename).suffix.lower() in RASTER_IMAGE_UPLOAD_SUFFIXES


def normalize_upload_to_pdf_bytes(*, filename: str, file_bytes: bytes) -> bytes:
    if not file_bytes:
        raise InputValidationError("Uploaded file is empty.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise InputValidationError(
            "Only PDF, PNG, and JPG uploads are supported.",
        )

    if suffix == ".pdf":
        return _read_pdf_bytes(file_bytes)

    return _image_bytes_to_pdf_bytes(file_bytes)


def _read_pdf_bytes(file_bytes: bytes) -> bytes:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            return document.tobytes()
    except Exception as error:
        raise InputValidationError(f"Failed to read uploaded PDF: {error}") from error


def _image_bytes_to_pdf_bytes(file_bytes: bytes) -> bytes:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
        image = image.convert("RGB")
    except Exception as error:
        raise InputValidationError(f"Failed to read uploaded image: {error}") from error

    width, height = image.size
    if width <= 0 or height <= 0:
        raise InputValidationError("Uploaded image has invalid dimensions.")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    jpeg_bytes = buffer.getvalue()

    document = fitz.open()
    try:
        page = document.new_page(width=width, height=height)
        page.insert_image(fitz.Rect(0, 0, width, height), stream=jpeg_bytes)
        return document.tobytes()
    except Exception as error:
        raise InputValidationError(f"Failed to convert uploaded image to PDF: {error}") from error
    finally:
        document.close()
