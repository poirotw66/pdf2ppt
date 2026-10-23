# Review UI

The project includes a lightweight review UI for OCR-heavy decks. Use it when
raw detection boxes need a human pass before inpainting and PPT generation.

If you only want CLI conversion, you do not need to run the review UI.

## Start the UI

Install the project into your active environment if you have not already
(see the root README). Then start the backend and frontend in two terminals:

```bash
# Terminal 1 — API
cd pdf2ppt
python -m pip install -e ".[ocr,server]"   # once
pdf2ppt-api
# or: python -m uvicorn pdf2ppt.api:app --host 127.0.0.1 --port 8008
```

```bash
# Terminal 2 — Vite frontend
cd pdf2ppt/frontend
npm install            # first time, or after dependency changes
npm run dev
```

Docker Compose alternative (API + static UI, named volumes for jobs/models, healthchecks; no `./output` bind — download PPTX over HTTP):

```bash
docker compose up --build
# UI http://127.0.0.1:8080  — optional GEMINI_API_KEY via .env (never baked into images)
```

Open `http://127.0.0.1:5173` in the browser. The Vite dev server proxies
`/jobs/*` to the FastAPI backend on port `8008`.

Notes:

- `npm install` must be run inside `frontend/`, not the repository root or home directory.
- If the same Conda environment also contains `iopaint` or older `gradio` / `fastapi` stacks, dependency conflicts can break API startup. Reinstall `pydantic`, `pydantic-core`, `fastapi`, and `starlette`, or use a dedicated environment for `pdf2ppt`.

## Review workflow

1. Upload a PDF in the web UI to create a job.
2. Run OCR detect to generate candidate text boxes and preview URLs.
3. Review each page and adjust the approved boxes before conversion.
4. Save the approved boxes back to the backend.
5. Run conversion so the pipeline uses approved boxes first, fills missing text with per-box recognition when needed, then generates the PPTX and report.

### Preview before convert

Optional second mode (no Gemini / layout planner required):

1. After upload, switch to **Preview before convert** and click **Build Preview** (`POST /jobs/{id}/preview`).
2. Inspect per-page source images with SlideIR overlays (text roles / reading order, tables, images, shapes). Toggle layers, exclude elements, change text roles, or mark a page as raster fallback (`PUT /jobs/{id}/preview-edits`).
3. Click **Convert** (`POST /jobs/{id}/from-ir`) to apply edits and download the PPTX built from the edited IR.

Notes:

- Re-running **Build Preview** or **emit-ir** deletes `preview_edits.json` for that job (element ids are positional and go stale after OCR box edits). The UI also resets its local edit state.
- Contended `PUT /preview-edits` and `POST /from-ir` wait up to ~**5s** for the job lock, with at most **4** waiters per job, then return `409` with code `job-busy`. The frontend auto-retries **preview-edits** (and related preview fetches) with **1s → 2s → 4s** backoff; upload, detect, final convert, and from-ir Convert do **not** auto-retry.
- `PUT /preview-edits` bodies larger than **1 MiB** return **413** (CORS headers still included).
- Exclude removes native SlideIR objects. On `overlay` / `full-page` pages (`baked_background: true`), content painted into the background raster may stay visible — if the preview does not change, use raster fallback for a full-page image.
- Thumbnails use `GET /jobs/{id}/pages/{n}.jpg?width=120` (`width` allowed range 16–4096).
- Rotated PDF pages may show overlay/image mismatch (element boxes use unrotated coordinates; pre-existing on `main`); prefer raster fallback there.

## Job storage and cleanup

- Uploaded PDFs, OCR detection payloads, approved box payloads, and conversion outputs are stored under `.pdf2ppt_jobs/` by default.
- Page previews are not persisted as artifacts. `GET /jobs/{job_id}/pages/{page_number}.jpg` renders the preview on demand from the uploaded PDF and returns JPEG.
- Automatic cleanup runs when a new job is created and removes job directories whose `updated_at` is older than the configured retention window.
- Configure the job root with `PDF2PPT_JOB_ROOT=/path/to/jobs`.
- Configure retention with `PDF2PPT_JOB_RETENTION_HOURS=24`. Set it to `0` to disable automatic cleanup.
- Optional browser CORS origins: `PDF2PPT_CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080` (defaults to the Vite ports). Comma-/whitespace-only values fall back to defaults. Do not set `*` — credentials are enabled, so reflecting any origin is rejected.
- `GEMINI_API_KEY` / `OPENAI_API_KEY` are optional and only used for the layout planner / generative engines.
- Liveness: `GET /health` → `{"status":"ok"}`.
- Without `[server]`, `pdf2ppt-api` exits with a short install hint (`pip install 'pdf2ppt[server]'`) instead of a raw `ModuleNotFoundError`.
- Use `DELETE /jobs/{job_id}` to remove a single job and all of its stored artifacts immediately.

## Editor capabilities

- Create a new box by dragging on empty canvas.
- Move an existing box by dragging it.
- Resize a selected box from its four corner handles.
- Use zoom, numeric bbox fields, and nudge buttons for fine correction.
- Edit box text, source, and confidence in the inspector.
- Duplicate or delete the selected box.
- Leave text empty for manually added boxes when you want convert-time OCR recognition to populate it.

## Backend endpoints

- `POST /jobs`
- `POST /jobs/{job_id}/detect` with optional `ocr_batch_size` and `confidence_threshold`
- `PUT /jobs/{job_id}/boxes`
- `POST /jobs/{job_id}/convert` with optional `ocr_batch_size`
- `POST /jobs/{job_id}/preview` — analyze → SlideIR (planner forced off) + overlay preview
- `GET /jobs/{job_id}/preview` — overlay preview from existing SlideIR + edits
- `PUT /jobs/{job_id}/preview-edits` — save exclude / role / raster-fallback edits
- `POST /jobs/{job_id}/from-ir` — apply edits and render PPTX from job SlideIR
- `GET /jobs/{job_id}/pages/{page_number}.jpg` optional `?width=` (16–4096)
- `GET /jobs/{job_id}/slideir.json` / `GET /jobs/{job_id}/slideir-assets.zip`
- `POST /jobs/{job_id}/emit-ir`
- `POST /from-ir` (multipart `ir_file` + optional `assets_zip`)
- `GET /jobs/{job_id}/output.pptx`
- `GET /jobs/{job_id}/report.json`
- `GET /health`
- `DELETE /jobs/{job_id}`

Detect request options:

- `confidence_threshold` filters out OCR boxes below the given confidence before the response is returned (default `0.75`).
- `return_word_box`: mirrors CLI `--ocr-return-word-box`, default `false`.
- The API applies the same line-merge post-processing as the CLI before returning boxes. Restart the API after code changes and rerun detect; existing job detection payloads are not refreshed automatically.

## Error response model

API errors return structured JSON under `detail`:

```json
{ "detail": { "code": "string", "message": "string", "page": null } }
```

`page` is only present for page-scoped conversion failures.

Common error codes:

- `input-error`: invalid upload, missing approved boxes, malformed approved box payload
- `ocr-initialization-error`: OCR runtime is unavailable or misconfigured
- `ocr-processing-error`: OCR runtime failed while processing the request
- `page-conversion-error`: a page failed during conversion
- `not-found`: unknown job id or missing generated artifact
- `job-busy`: another request is still working on this job (`409`; max 4 waiters, ~5s lock wait)

Typical endpoint status codes:

- `POST /jobs`: `200`, `400`
- `POST /jobs/{job_id}/detect`: `200`, `404`, `502`, `503`
- `PUT /jobs/{job_id}/boxes`: `200`, `400`, `404`
- `POST /jobs/{job_id}/convert`: `200`, `400`, `404`, `500`, `502`, `503`
- `POST /jobs/{job_id}/preview`: `200`, `404`, `500`, `502`, `503`
- `PUT /jobs/{job_id}/preview-edits`: `200`, `400`, `404`, `409`, `413`
- `POST /jobs/{job_id}/from-ir`: `200`, `400`, `404`, `409`, `500`
- `POST /from-ir`: `200`, `400` (also zip safety / size limits)
- `GET /health`: `200`

## Related source files

- API / review backend: `src/pdf2ppt/api.py`
- OCR and line merging: `src/pdf2ppt/ocr.py`
- Frontend: `frontend/`
