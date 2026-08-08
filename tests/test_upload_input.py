from __future__ import annotations

import io
import unittest

import fitz
from PIL import Image

from pdf2ppt.core import InputValidationError
from pdf2ppt.upload_input import is_raster_image_upload, normalize_upload_to_pdf_bytes


class UploadInputTests(unittest.TestCase):
    def test_is_raster_image_upload_detects_png_and_jpg(self) -> None:
        self.assertTrue(is_raster_image_upload("slide.PNG"))
        self.assertTrue(is_raster_image_upload("photo.jpg"))
        self.assertFalse(is_raster_image_upload("deck.pdf"))

    def test_normalize_upload_rejects_empty_file(self) -> None:
        with self.assertRaises(InputValidationError):
            normalize_upload_to_pdf_bytes(filename="slide.png", file_bytes=b"")

    def test_normalize_upload_rejects_unsupported_suffix(self) -> None:
        with self.assertRaises(InputValidationError):
            normalize_upload_to_pdf_bytes(filename="notes.txt", file_bytes=b"plain-text")

    def test_normalize_upload_converts_png_to_single_page_pdf(self) -> None:
        image = Image.new("RGB", (180, 120), color=(30, 40, 50))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        pdf_bytes = normalize_upload_to_pdf_bytes(filename="slide.png", file_bytes=buffer.getvalue())

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            self.assertEqual(document.page_count, 1)
            page = document[0]
            self.assertAlmostEqual(page.rect.width, 180, delta=1)
            self.assertAlmostEqual(page.rect.height, 120, delta=1)
