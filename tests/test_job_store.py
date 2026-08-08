from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pdf2ppt.job_store import JobStore


class JobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.job_store = JobStore(Path(self.temp_dir.name) / "jobs")

    def test_delete_expired_jobs_removes_stale_job_directories(self) -> None:
        stale_record = self.job_store.create_job(filename="stale.pdf", pdf_bytes=build_sample_pdf_bytes())
        fresh_record = self.job_store.create_job(filename="fresh.pdf", pdf_bytes=build_sample_pdf_bytes())
        stale_updated_at = datetime.now(UTC) - timedelta(hours=48)
        metadata_path = self.job_store.metadata_path(stale_record.job_id)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["updated_at"] = stale_updated_at.isoformat()
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        deleted_count = self.job_store.delete_expired_jobs(now=datetime.now(UTC))

        self.assertEqual(deleted_count, 1)
        self.assertFalse(self.job_store.job_dir(stale_record.job_id).exists())
        self.assertTrue(self.job_store.job_dir(fresh_record.job_id).exists())

    def test_delete_job_removes_job_directory(self) -> None:
        record = self.job_store.create_job(filename="sample.pdf", pdf_bytes=build_sample_pdf_bytes())

        self.job_store.delete_job(record.job_id)

        self.assertFalse(self.job_store.job_dir(record.job_id).exists())
        with self.assertRaises(FileNotFoundError):
            self.job_store.get_job(record.job_id)

    def test_zero_retention_disables_automatic_cleanup(self) -> None:
        self.job_store.retention = None
        stale_record = self.job_store.create_job(filename="stale.pdf", pdf_bytes=build_sample_pdf_bytes())

        deleted_count = self.job_store.delete_expired_jobs(now=datetime.now(UTC))

        self.assertEqual(deleted_count, 0)
        self.assertTrue(self.job_store.job_dir(stale_record.job_id).exists())


def build_sample_pdf_bytes() -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj\n"
        b"trailer << /Root 1 0 R >>\n%%EOF\n"
    )
