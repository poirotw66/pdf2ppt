# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-10-23

### Added

- Optional packaging extras: `[ocr]`, `[server]`, `[dev]`, `[all]`, plus existing `[cpu]` / `[gpu]` / `[lint]`.
- Explicit `paddlepaddle==3.2.0` pin under `[ocr]` (3.3.1 breaks PP-OCRv5 OCR on CPU).
- `python -m pdf2ppt` module entry (via `__main__`) documented alongside the `pdf2ppt` console script.
- Docker support: `Dockerfile`, `Dockerfile.frontend`, `docker-compose.yml`, `.dockerignore`, `.env.example`.
- API `GET /health` liveness endpoint for container healthchecks.
- `PDF2PPT_CORS_ORIGINS` env override for Review UI origins (defaults unchanged).
- Makefile shortcuts (`install`, `lint`, `test`, `build-wheel`, `verify-wheel`, `docker-lint`, …).
- Quick-start docs: `docs/quickstart.md`; README / README_tw install and Docker sections refreshed.
- CI packaging job: build sdist+wheel and smoke-test the wheel in a clean venv.

### Changed

- Core `pip install .` no longer pulls PaddleOCR or FastAPI by default; install `pdf2ppt[ocr]` / `pdf2ppt[server]` (or `[dev]` / `[all]`) as needed. Conversion algorithms and PPTX output are unchanged.
- Missing OCR on scanned pages prints a clean `error:` line (exit 3) that mentions `pip install 'pdf2ppt[ocr]'` instead of a raw traceback.
- `pdf2ppt-api` without `[server]` exits 3 with an install hint (via `pdf2ppt.api_entry`) instead of `ModuleNotFoundError: uvicorn`.
- Docker image sets `PDF2PPT_ROOT=/data` so OCR models cache on the `/data/model` volume (not `/app/model`).
- `PDF2PPT_CORS_ORIGINS` of only commas / whitespace falls back to Vite defaults; `*` is rejected while credentials are enabled.
- Compose drops the unused `./output` bind mount; OCR models default to the named volume `pdf2ppt-models` (uid 10001). Host bind mounts via `PDF2PPT_MODEL_DIR` need `chown 10001:10001`.

## [0.1.0] - 2026-10-22

### Added

- Initial hybrid PDF → editable PPTX pipeline, Review UI, SlideIR, and CI.
