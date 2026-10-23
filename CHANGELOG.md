# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `pdf2ppt batch <input_dir> -o <output_dir>`: folder batch conversion with shared single-file options, failure isolation, `--fail-fast`, `--skip-existing` / `--resume` (retain prior ok metrics; validate pptx zip), per-file atomic summary flush + atomic `.pptx.partial` writes, optional `--jobs` (default 1; process pool; OCR memory warning), quality summaries (`summary.csv` / `summary.json` / `summary.md`), and optional `--measure-quality` (LibreOffice SSIM, skipped if missing). Exit codes **0** / **1** / **2**; KeyboardInterrupt exits **130** with a valid partial summary — rerun with `--resume`. SIGINT during native OCR may still yield a non-zero abnormal interpreter exit; `summary.*` remains usable (#27).
- Preview before convert in the Review UI: `POST/GET /jobs/{id}/preview`, `PUT /jobs/{id}/preview-edits` (body cap **1 MiB** → **413**), `POST /jobs/{id}/from-ir`, thumbnails `GET …/pages/{n}.jpg?width=` (UI uses 120), `baked_background` warning on overlay/full-page pages, re-analysis / emit-ir clears edits (#24, #25).
- Job lock for contended preview-edits / from-ir: ~5s wait, max **4** waiters, then **409** `job-busy`; frontend auto-retries preview-edits with **1s / 2s / 4s** backoff (not upload / final convert / from-ir) (#25, #26).
- Hardening: safe `assets_zip` limits for `POST /from-ir` (`PDF2PPT_ZIP_MAX_*`, defaults 50 / 200 MiB, 10k entries, 50 MiB/entry, ratio 100); per-page OCR/layout timeout `--page-timeout-s` / `PDF2PPT_PAGE_TIMEOUT_S` (default **120s**, `<=0` disables) with OCR timeout-lane degrade; layout-planner circuit cancels reserved USD calls when opened (#23, #25).

### Documentation

- README / README_tw restructured (features, quick start, batch, Review UI, env table, known limitations, FAQ); `docs/README.md` index; env template and docs synced to CLI/API defaults (#14–#27 wrap-up).

## [0.1.1] - 2026-10-23

### Added

- Optional packaging extras: `[ocr]`, `[server]`, `[dev]`, `[all]`, plus existing `[cpu]` / `[gpu]` / `[lint]` / `[layout]`.
- Explicit `paddlepaddle==3.2.0` pin under `[ocr]` (3.3.x breaks PP-OCRv5 on CPU; use 3.2.0).
- `python -m pdf2ppt` module entry (via `__main__`) documented alongside the `pdf2ppt` console script.
- Docker support: `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`, `.dockerignore`, `.env.example`.
- API `GET /health` liveness endpoint for container healthchecks.
- `PDF2PPT_CORS_ORIGINS` env override for Review UI origins (defaults unchanged).
- Makefile shortcuts (`install`, `lint`, `test`, `build-wheel`, `verify-wheel`, `docker-lint`, …).
- Quick-start docs: `docs/quickstart.md`; README / README_tw install and Docker sections refreshed.
- CI packaging job: build sdist+wheel and smoke-test the wheel in a clean venv (#22).

### Changed

- Core `pip install .` no longer pulls PaddleOCR or FastAPI by default; install `pdf2ppt[ocr]` / `pdf2ppt[server]` (or `[dev]` / `[all]`) as needed. Conversion algorithms and PPTX output are unchanged.
- Missing OCR on scanned pages prints a clean `error:` line (exit 3) that mentions `pip install 'pdf2ppt[ocr]'` instead of a raw traceback.
- `pdf2ppt-api` without `[server]` exits 3 with an install hint (via `pdf2ppt.api_entry`) instead of `ModuleNotFoundError: uvicorn`.
- Docker image sets `PDF2PPT_ROOT=/data` so OCR models cache on the `/data/model` volume (not `/app/model`).
- `PDF2PPT_CORS_ORIGINS` of only commas / whitespace falls back to Vite defaults; `*` is rejected while credentials are enabled.
- Compose drops the unused `./output` bind mount; OCR models default to the named volume `pdf2ppt-models` (uid 10001). Host bind mounts via `PDF2PPT_MODEL_DIR` need `chown 10001:10001`.

## [0.1.0] - 2026-10-22

### Added

- Initial hybrid PDF → editable PPTX pipeline, Review UI, SlideIR (`emit-ir` / `from-ir`), and CI.
- Optional deterministic render-compare-refine loop (`--refine`): LibreOffice SSIM-gated SlideIR fixes, font floor/ratio, attempt cap, fair time budget, severity ordering, leftover-time reallocation (#14–#17).
- Experimental opt-in Gemini VLM layout planner (`--layout-planner gemini`): USD + call caps, circuit breaker, low-risk default apply; `--layout-planner-apply-grouping` / `--layout-planner-apply-fonts`; failed-call reserve refund (#18, #19).
- Borderless / zebra table inference, tables over photo underlays (transparent cells, no erase), tall form cells, table z-order (#21).
- SlideIR documented in README / README_tw with grouping-flag accuracy (#20).
