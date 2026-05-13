from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient

from pdf2ppt.api import app
from pdf2ppt.core import OcrInitializationError, PageConversionError
from pdf2ppt.job_store import JobStore


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.job_store = JobStore(Path(self.temp_dir.name) / "jobs")
        self.client = TestClient(app)
        self.job_store_patcher = patch("pdf2ppt.api.job_store", self.job_store)
        self.job_store_patcher.start()
        self.addCleanup(self.job_store_patcher.stop)

    def test_create_job_and_fetch_metadata(self) -> None:
        pdf_bytes = build_sample_pdf_bytes()

        create_response = self.client.post(
            "/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        self.assertEqual(payload["status"], "uploaded")
        self.assertEqual(payload["page_count"], 1)

        get_response = self.client.get(f"/jobs/{payload['job_id']}")
        self.assertEqual(get_response.status_code, 200)
        fetched = get_response.json()
        self.assertEqual(fetched["job_id"], payload["job_id"])
        self.assertEqual(fetched["original_filename"], "sample.pdf")

    @patch("pdf2ppt.api.OcrEngine.extract_text_blocks_batch")
    def test_detect_generates_preview_payload(
        self,
        extract_text_blocks_batch_mock: unittest.mock.Mock,
    ) -> None:
        extract_text_blocks_batch_mock.return_value = [
            SimpleNamespace(
                blocks=[
                    SimpleNamespace(
                        id="ocr_1_1",
                        bbox=(10.0, 20.0, 50.0, 60.0),
                        image_polygon=((10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)),
                        text="demo",
                        confidence=0.95,
                    )
                ]
            )
        ]
        pdf_bytes = build_sample_pdf_bytes()
        create_response = self.client.post(
            "/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        job_id = create_response.json()["job_id"]

        detect_response = self.client.post(f"/jobs/{job_id}/detect", json={"dpi": 110})
        self.assertEqual(detect_response.status_code, 200)
        extract_text_blocks_batch_mock.assert_called_once()
        self.assertEqual(extract_text_blocks_batch_mock.call_args.args[1], [1])
        payload = detect_response.json()
        self.assertEqual(payload["status"], "detected")
        self.assertEqual(len(payload["pages"]), 1)
        first_page = payload["pages"][0]
        self.assertEqual(first_page["page"], 1)
        self.assertEqual(first_page["boxes"][0]["source"], "ocr-auto")
        self.assertEqual(first_page["boxes"][0]["text"], "demo")

        preview_response = self.client.get(first_page["image_url"])
        self.assertEqual(preview_response.status_code, 200)
        self.assertEqual(preview_response.headers["content-type"], "image/png")

    def test_put_boxes_persists_approved_boxes(self) -> None:
        pdf_bytes = build_sample_pdf_bytes()
        create_response = self.client.post(
            "/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        job_id = create_response.json()["job_id"]

        payload = {
            "pages": [
                {
                    "page": 1,
                    "width": 320,
                    "height": 240,
                    "boxes": [
                        {
                            "id": "box_1",
                            "source": "ocr-user",
                            "bbox": [10.0, 20.0, 80.0, 60.0],
                            "polygon": [[10.0, 20.0], [80.0, 20.0], [80.0, 60.0], [10.0, 60.0]],
                            "text": "approved",
                            "confidence": 1.0,
                        }
                    ],
                }
            ]
        }

        response = self.client.put(f"/jobs/{job_id}/boxes", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "boxes-approved")
        self.assertEqual(body["pages"][0]["boxes"][0]["text"], "approved")

        job_response = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(job_response.status_code, 200)
        job_payload = job_response.json()
        self.assertIsNotNone(job_payload["approved_boxes_path"])
        approved_boxes_path = Path(job_payload["approved_boxes_path"])
        self.assertTrue(approved_boxes_path.exists())

    def test_create_job_returns_input_error_for_invalid_pdf(self) -> None:
        response = self.client.post(
            "/jobs",
            files={"file": ("broken.pdf", b"not-a-pdf", "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "input-error")

    @patch("pdf2ppt.api.convert_pdf")
    def test_convert_uses_approved_boxes_and_updates_job(
        self,
        convert_pdf_mock: unittest.mock.Mock,
    ) -> None:
        convert_pdf_mock.return_value = SimpleNamespace(pages=[object()])
        pdf_bytes = build_sample_pdf_bytes()
        create_response = self.client.post(
            "/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        job_id = create_response.json()["job_id"]
        boxes_payload = {
            "pages": [
                {
                    "page": 1,
                    "width": 320,
                    "height": 240,
                    "boxes": [
                        {
                            "id": "box_1",
                            "source": "ocr-user",
                            "bbox": [10.0, 20.0, 80.0, 60.0],
                            "text": "",
                            "confidence": 1.0,
                        }
                    ],
                }
            ]
        }
        self.client.put(f"/jobs/{job_id}/boxes", json=boxes_payload)

        response = self.client.post(f"/jobs/{job_id}/convert", json={"write_debug_artifacts": False})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "converted")
        self.assertEqual(body["page_count"], 1)

        options = convert_pdf_mock.call_args.args[0]
        self.assertIn(1, options.approved_ocr_blocks_by_page)
        self.assertEqual(options.approved_ocr_image_size_by_page[1], (320, 240))
        self.assertEqual(options.ocr_batch_size, 3)

        job_response = self.client.get(f"/jobs/{job_id}")
        job_payload = job_response.json()
        self.assertEqual(job_payload["status"], "converted")
        self.assertIsNotNone(job_payload["output_pptx_path"])
        self.assertIsNotNone(job_payload["report_path"])

    @patch("pdf2ppt.api.convert_pdf")
    def test_convert_accepts_custom_ocr_batch_size(
        self,
        convert_pdf_mock: unittest.mock.Mock,
    ) -> None:
        convert_pdf_mock.return_value = SimpleNamespace(pages=[object()])
        pdf_bytes = build_sample_pdf_bytes()
        create_response = self.client.post(
            "/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        job_id = create_response.json()["job_id"]
        self.client.put(
            f"/jobs/{job_id}/boxes",
            json={
                "pages": [
                    {
                        "page": 1,
                        "width": 320,
                        "height": 240,
                        "boxes": [],
                    }
                ]
            },
        )

        response = self.client.post(
            f"/jobs/{job_id}/convert",
            json={"write_debug_artifacts": False, "ocr_batch_size": 6},
        )

        self.assertEqual(response.status_code, 200)
        options = convert_pdf_mock.call_args.args[0]
        self.assertEqual(options.ocr_batch_size, 6)

    @patch("pdf2ppt.api.OcrEngine.extract_text_blocks_batch")
    def test_detect_returns_ocr_initialization_error(
        self,
        extract_text_blocks_batch_mock: unittest.mock.Mock,
    ) -> None:
        extract_text_blocks_batch_mock.side_effect = OcrInitializationError("missing OCR runtime")
        pdf_bytes = build_sample_pdf_bytes()
        create_response = self.client.post(
            "/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        job_id = create_response.json()["job_id"]

        response = self.client.post(f"/jobs/{job_id}/detect", json={"dpi": 110})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "ocr-initialization-error")

    @patch("pdf2ppt.api.convert_pdf")
    def test_convert_returns_page_conversion_error_detail(
        self,
        convert_pdf_mock: unittest.mock.Mock,
    ) -> None:
        convert_pdf_mock.side_effect = PageConversionError(2, "failed to process page 2")
        pdf_bytes = build_sample_pdf_bytes()
        create_response = self.client.post(
            "/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        job_id = create_response.json()["job_id"]
        self.client.put(
            f"/jobs/{job_id}/boxes",
            json={
                "pages": [
                    {
                        "page": 1,
                        "width": 320,
                        "height": 240,
                        "boxes": [],
                    }
                ]
            },
        )

        response = self.client.post(f"/jobs/{job_id}/convert", json={"write_debug_artifacts": False})

        self.assertEqual(response.status_code, 500)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "page-conversion-error")
        self.assertEqual(detail["page"], 2)

    def test_download_endpoints_serve_existing_outputs(self) -> None:
        pdf_bytes = build_sample_pdf_bytes()
        create_response = self.client.post(
            "/jobs",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
        job_id = create_response.json()["job_id"]
        job_dir = self.job_store.job_dir(job_id)
        output_path = job_dir / "output.pptx"
        report_path = job_dir / "output.report.json"
        output_path.write_bytes(b"pptx-bytes")
        report_path.write_text('{"ok": true}', encoding="utf-8")
        self.job_store.update_job(
            job_id,
            status="converted",
            output_pptx_path=str(output_path),
            report_path=str(report_path),
        )

        output_response = self.client.get(f"/jobs/{job_id}/output.pptx")
        self.assertEqual(output_response.status_code, 200)
        self.assertEqual(output_response.content, b"pptx-bytes")

        report_response = self.client.get(f"/jobs/{job_id}/report.json")
        self.assertEqual(report_response.status_code, 200)
        self.assertEqual(report_response.json(), {"ok": True})


def build_sample_pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page(width=320, height=240)
    page.insert_text((50, 80), "Hello API")
    return document.tobytes()
