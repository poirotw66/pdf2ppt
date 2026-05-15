from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import fitz


DEFAULT_JOB_ROOT = Path(".pdf2ppt_jobs")
DEFAULT_JOB_RETENTION_HOURS = 24


@dataclass(slots=True)
class JobRecord:
    job_id: str
    status: str
    original_filename: str
    input_pdf_path: str
    page_count: int
    created_at: str
    updated_at: str
    detection_path: str | None = None
    approved_boxes_path: str | None = None
    output_pptx_path: str | None = None
    report_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "JobRecord":
        return cls(**payload)


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        configured_root = os.environ.get("PDF2PPT_JOB_ROOT")
        self.root = (root or Path(configured_root) if configured_root else root or DEFAULT_JOB_ROOT).resolve()
        retention_hours = os.environ.get("PDF2PPT_JOB_RETENTION_HOURS")
        configured_retention_hours = int(retention_hours or DEFAULT_JOB_RETENTION_HOURS)
        self.retention = timedelta(hours=configured_retention_hours) if configured_retention_hours > 0 else None
        self.root.mkdir(parents=True, exist_ok=True)

    def create_job(self, *, filename: str, pdf_bytes: bytes) -> JobRecord:
        self.delete_expired_jobs()
        job_id = uuid4().hex
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        input_pdf_path = job_dir / "input.pdf"
        input_pdf_path.write_bytes(pdf_bytes)
        with fitz.open(input_pdf_path) as document:
            page_count = document.page_count
        record = JobRecord(
            job_id=job_id,
            status="uploaded",
            original_filename=filename,
            input_pdf_path=str(input_pdf_path),
            page_count=page_count,
            created_at=_utc_now_iso(),
            updated_at=_utc_now_iso(),
        )
        self.save_job(record)
        return record

    def get_job(self, job_id: str) -> JobRecord:
        metadata_path = self.metadata_path(job_id)
        if not metadata_path.exists():
            raise FileNotFoundError(job_id)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return JobRecord.from_dict(payload)

    def save_job(self, record: JobRecord) -> JobRecord:
        record.updated_at = _utc_now_iso()
        metadata_path = self.metadata_path(record.job_id)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def update_job(self, job_id: str, **changes: object) -> JobRecord:
        record = self.get_job(job_id)
        for key, value in changes.items():
            setattr(record, key, value)
        return self.save_job(record)

    def save_detection_payload(self, job_id: str, payload: dict[str, object]) -> Path:
        output_path = self.job_dir(job_id) / "detection.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.update_job(job_id, detection_path=str(output_path), status="detected")
        return output_path

    def save_approved_boxes_payload(self, job_id: str, payload: dict[str, object]) -> Path:
        output_path = self.job_dir(job_id) / "approved_boxes.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.update_job(job_id, approved_boxes_path=str(output_path), status="boxes-approved")
        return output_path

    def delete_job(self, job_id: str) -> None:
        job_dir = self.job_dir(job_id)
        if not job_dir.exists():
            raise FileNotFoundError(job_id)
        shutil.rmtree(job_dir)

    def delete_expired_jobs(self, *, now: datetime | None = None) -> int:
        if self.retention is None:
            return 0
        expiration_time = (now or datetime.now(timezone.utc)) - self.retention
        deleted_count = 0
        for job_dir in self.root.iterdir():
            if not job_dir.is_dir():
                continue
            metadata_path = job_dir / "job.json"
            if not metadata_path.exists():
                shutil.rmtree(job_dir)
                deleted_count += 1
                continue
            try:
                record = JobRecord.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
                updated_at = datetime.fromisoformat(record.updated_at)
            except Exception:
                shutil.rmtree(job_dir)
                deleted_count += 1
                continue
            if updated_at <= expiration_time:
                shutil.rmtree(job_dir)
                deleted_count += 1
        return deleted_count

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def metadata_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def preview_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "previews"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def preview_image_path(self, job_id: str, page_number: int) -> Path:
        return self.preview_dir(job_id) / f"page_{page_number:03d}.png"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
