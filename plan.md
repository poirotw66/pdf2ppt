# Frontend + Backend Plan

## Goal

Build a frontend/backend workflow on top of the existing `pdf2ppt` pipeline so users can:

1. Upload a PDF.
2. Run OCR detection first.
3. Review detected boxes on the frontend.
4. Delete incorrect boxes.
5. Add missing boxes.
6. Submit the approved boxes.
7. Run the downstream recognition, inpaint, and PPTX conversion flow only after box approval.

## Product Flow

1. User uploads a PDF.
2. Backend creates a job and stores the source PDF.
3. Backend rasterizes pages and runs OCR detection preview.
4. Frontend displays page images with editable OCR boxes.
5. User edits the boxes.
6. Frontend submits approved boxes back to the backend.
7. Backend uses approved boxes for the later conversion stages.
8. Backend generates PPTX and JSON report.
9. Frontend polls job status and provides download links.

## Backend Design

### Phase 1: Minimal FastAPI backend

Implement a minimal backend that supports:

- `POST /jobs`
  - Accept PDF upload.
  - Create a job id.
  - Persist the input PDF under a local job workspace.

- `POST /jobs/{job_id}/detect`
  - Render PDF pages to preview images.
  - Run OCR detection/preview for each page.
  - Return page image URLs plus editable box payload.

- `GET /jobs/{job_id}`
  - Return job metadata and current status.

- `GET /jobs/{job_id}/pages/{page_number}.png`
  - Serve rendered page preview images.

### Phase 2: User-reviewed box workflow

Implement:

- `PUT /jobs/{job_id}/boxes`
  - Save user-approved boxes.
  - Allow delete, move, resize, and manual box creation from frontend edits.

### Phase 3: Conversion after approval

Implement:

- `POST /jobs/{job_id}/convert`
  - Use approved boxes instead of raw OCR candidate boxes.
  - Run recognition, style estimation, mask generation, inpaint, and PPTX output.

## Data Model Direction

### Job

Suggested job fields:

- `job_id`
- `status`
- `input_pdf_path`
- `page_count`
- `created_at`
- `updated_at`
- `approved_boxes_path`
- `report_path`
- `output_pptx_path`

### OCR box payload

Suggested response shape:

```json
{
  "job_id": "...",
  "pages": [
    {
      "page": 1,
      "image_url": "/jobs/.../pages/1.png",
      "width": 1376,
      "height": 768,
      "boxes": [
        {
          "id": "ocr_1_1",
          "source": "ocr-auto",
          "bbox": [59.0, 52.5, 881.0, 172.5],
          "polygon": [[59.0, 58.0], [880.5, 52.5], [881.0, 167.0], [59.5, 172.5]],
          "text": null,
          "confidence": 0.94
        }
      ]
    }
  ]
}
```

## Integration with Existing Code

### Reuse

- OCR engine: `src/pdf2ppt/ocr.py`
- Text block model: `src/pdf2ppt/models.py`
- Page rendering helpers: current PDF/page pipeline modules
- Downstream inpaint and PPT conversion: `src/pdf2ppt/pipeline.py`

### Required changes later

- Split OCR preview/detection from later recognition if needed.
- Add a way for downstream conversion to accept approved OCR blocks.
- Distinguish `ocr-auto` vs `ocr-user` sources in report/debug payloads.

## Frontend Design Direction

Recommended stack:

- React + Vite
- Page image preview from backend-rendered PNGs
- Box editing canvas with Konva or Fabric.js

Main UI regions:

- Left: page list / thumbnails
- Center: page preview with editable boxes
- Right: selected box controls and actions

Core interactions:

- select box
- delete box
- drag box
- resize box
- add rectangle

## Coordinate Strategy

Use page preview image pixel coordinates end-to-end between frontend and review APIs.

Reason:

- easier canvas editing
- fewer browser/layout conversion bugs
- mapping back to PDF/page coordinates can happen once in backend conversion logic

## Initial Implementation Scope

This implementation pass will start with:

1. `plan.md`
2. FastAPI dependency wiring
3. Minimal job storage
4. `POST /jobs`
5. `GET /jobs/{job_id}`
6. `POST /jobs/{job_id}/detect`
7. `GET /jobs/{job_id}/pages/{page_number}.png`

The box editing save API and final conversion API will follow after the scaffold is stable.
