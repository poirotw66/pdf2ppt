# pdf2ppt

[![CI](https://github.com/poirotw66/pdf2ppt/actions/workflows/ci.yml/badge.svg)](https://github.com/poirotw66/pdf2ppt/actions/workflows/ci.yml)

**English** | [繁體中文](README_tw.md)

Convert PDF slides into **editable** PowerPoint (`.pptx`). Native extraction
first, OCR for scanned pages, optional layout / VLM planning, and backgrounds
only where text was removed.

## Showcase

<p align="center">
  <img src="image/example_pdf.png" alt="Example source PDF slide" width="48%" />
  <img src="image/example_ppt.png" alt="Example converted PPT slide" width="48%" />
</p>

<p align="center"><em>Source PDF on the left, converted editable PowerPoint on the right.</em></p>

## Features

- Native PDF extraction; PaddleOCR fallback for scanned / image-heavy pages (`[ocr]`)
- Optional layout analysis (heuristic / PP-Structure) and experimental Gemini layout planner (USD + call caps, circuit breaker, `--layout-planner-apply-grouping`)
- Vector / table nativization: ruled tables, **borderless / zebra** inference, **tables over photos** ([`docs/vector-nativization.md`](docs/vector-nativization.md))
- SlideIR agent layer: `emit-ir` / `from-ir` + HTTP (`POST /jobs`, `/emit-ir`, `/slideir.json`, `/slideir-assets.zip`, `POST /from-ir`)
- Optional `--refine` render-compare loop (LibreOffice)
- Review UI with **Preview before convert** (overlays, light edits, thumbnails, `baked_background`)
- Hardening: safe zip limits for `from-ir`, per-page timeouts (`PDF2PPT_PAGE_TIMEOUT_S`, default **120s**), OCR timeout-lane degrade, planner circuit breaker
- Batch folder convert: `pdf2ppt batch` with resume, summaries, exit codes **0 / 1 / 2**
- Packaging: pip / pipx extras `[ocr]` `[server]` `[dev]` `[all]` (paddlepaddle **3.2.0**), `python -m pdf2ppt`, Docker Compose

Docs index: [`docs/README.md`](docs/README.md).

## Quick start

Requirements: Python **3.11+**, Linux recommended. Keep `numpy<2` with OCR.

### pip / venv

```bash
cd pdf2ppt
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[ocr]"          # conversion + OCR
# python -m pip install -e ".[dev]"    # editable: OCR + API + lint/pytest
pdf2ppt tests/fixtures/solid_background.pdf output.pptx
# same entry: python -m pdf2ppt …
```

### pipx

```bash
pipx install ".[ocr]"                   # from a local clone
pdf2ppt --help
```

### Extras

| Extra | What you get |
| --- | --- |
| *(core)* | Digital/native PDF → PPTX |
| `[ocr]` | PaddleOCR + **paddlepaddle==3.2.0** (3.3.x breaks PP-OCRv5 on CPU) |
| `[server]` | FastAPI Review UI backend (`pdf2ppt-api`) |
| `[dev]` | `[ocr]` + `[server]` + ruff + mypy + pytest |
| `[all]` | `[ocr]` + `[server]` + CPU ONNX Runtime |
| `[cpu]` / `[gpu]` | ONNX Runtime for `lama-onnx` |
| `[layout]` | Alias of `[ocr]` (PP-Structure layout) |

OCR models default to `./model` (`--ocr-model-root`). Prefer `pip install -e ".[dev]"` for development; do not point `PYTHONPATH` at `build/lib/`.

### Docker

```bash
docker compose up --build
# Review UI: http://127.0.0.1:8080
# API health: http://127.0.0.1:8008/health
```

Image sets `PDF2PPT_ROOT=/data`. OCR models use named volume `pdf2ppt-models`
(writable by container **uid 10001**). For a host bind: `PDF2PPT_MODEL_DIR=./model`
and `chown 10001:10001`. Jobs use `pdf2ppt-jobs` (no `./output` bind).

```bash
docker build -t pdf2ppt .
# Match host uid so bind-mount writes succeed (image USER is 10001).
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -w /work pdf2ppt \
  pdf2ppt tests/fixtures/solid_background.pdf /work/output.pptx
```

`GEMINI_API_KEY` is optional (planner / generative Gemini only) and runtime-only
(see `.env.example`). Rebuild with `--build-arg INSTALL_LIBREOFFICE=1` for `--refine`.

## Basic usage

```bash
pdf2ppt input.pdf output.pptx --report output.report.json --debug-dir output_debug
pdf2ppt input.pdf output.pptx --inpaint-engine opencv-fast

# Experimental refine (LibreOffice; skipped with a warning if soffice missing)
pdf2ppt input.pdf output.pptx --refine --refine-iters 2

# Experimental Gemini layout planner (GEMINI_API_KEY; per-page fallback)
pdf2ppt input.pdf output.pptx --layout-planner gemini
pdf2ppt input.pdf output.pptx --layout-planner gemini --layout-planner-apply-grouping

# SlideIR round-trip
pdf2ppt emit-ir input.pdf deck.slideir.json
pdf2ppt from-ir deck.slideir.json rebuilt.pptx
```

Full flag list: [`docs/cli-reference.md`](docs/cli-reference.md).

### SlideIR (short)

Schema `1.0`; assets live next to the IR as `{ir}.assets/`. Planner roles /
reading order apply when `--layout-planner gemini` succeeds; **`groups` only
with `--layout-planner-apply-grouping`**. HTTP (API on `:8008`):

| Endpoint | Behaviour |
| --- | --- |
| `POST /jobs` | Upload PDF/PNG/JPG → job |
| `POST /jobs/{id}/emit-ir` | Analysis → SlideIR |
| `GET /jobs/{id}/slideir.json` | Download IR |
| `GET /jobs/{id}/slideir-assets.zip` | Download assets zip |
| `POST /from-ir` | Multipart `ir_file` + optional `assets_zip` → PPTX |

Details and zip safety limits: [`docs/slideir.md`](docs/slideir.md).

## Batch usage

```bash
pdf2ppt batch ./pdfs -o ./pptx_out
pdf2ppt batch ./pdfs -o ./pptx_out --recursive --resume --skip-existing
pdf2ppt batch ./pdfs -o ./pptx_out --fail-fast --jobs 1 --measure-quality
```

| Topic | Behaviour |
| --- | --- |
| Discovery | `*.pdf` / `*.PDF`; non-recursive by default; `--recursive` mirrors subfolders; skips hidden names and PDFs already under the output dir |
| Isolation | One bad PDF does not stop the batch (unless `--fail-fast`) |
| Resume | `--skip-existing` / `--resume` keep valid PPTX; after interrupt, rerun with `--resume` |
| Summaries | `summary.csv` / `summary.json` / `summary.md` (atomic flush after each file) |
| Jobs | `--jobs` default **1**; `>1` uses processes (OCR is memory-heavy) |
| Exit codes | **0** all ok/skipped · **1** ≥1 failed · **2** usage/validation · **130** KeyboardInterrupt (partial summary kept) |

Per-file PPTX matches single-file conversion aside from known PaddleOCR warm-up
colour drift (~±2/255) when several OCR-heavy files share one process. **Note:**
SIGINT during native OCR can still end with a non-zero abnormal code from the
interpreter; `summary.*` remains valid for `--resume`.

## Review UI / API overview

```bash
# Terminal 1
python -m pip install -e ".[ocr,server]"
pdf2ppt-api    # http://127.0.0.1:8008

# Terminal 2
cd frontend && npm install && npm run dev   # http://127.0.0.1:5173
```

Or `docker compose up --build` → UI on **8080**.

**Preview before convert:** Build Preview → edit overlays (exclude / role /
raster fallback) → Convert via `from-ir`.

| Topic | Detail |
| --- | --- |
| Endpoints | `POST/GET …/preview`, `PUT …/preview-edits`, `POST …/from-ir`, page thumbs `GET …/pages/{n}.jpg?width=` (UI uses **120**) |
| Planner | Preview forces layout planner **off** (no Gemini key) |
| `baked_background` | `true` on `overlay` / `full-page`; exclude may leave baked pixels — use raster fallback |
| **413** | Preview-edits body > **1 MiB** |
| **409 `job-busy`** | Job lock wait ~5s, max **4** waiters; UI auto-retries preview-edits with **1s / 2s / 4s** (not upload / final convert / from-ir) |

Full workflow: [`docs/review-ui.md`](docs/review-ui.md).

## Configuration (environment)

| Variable | Default | Purpose |
| --- | --- | --- |
| `PDF2PPT_ROOT` | auto (repo) / Docker `/data` | Project root for models / relative paths |
| `PDF2PPT_JOB_ROOT` | `.pdf2ppt_jobs` | Review UI job store |
| `PDF2PPT_JOB_RETENTION_HOURS` | `24` | Job TTL (`<=0` disables expiry) |
| `PDF2PPT_CORS_ORIGINS` | Vite `5173` origins | Comma-separated; `*` rejected (credentials on); comma-only → defaults |
| `PDF2PPT_PAGE_TIMEOUT_S` | `120` | Per-page OCR/layout timeout (`<=0` disables) |
| `PDF2PPT_ZIP_MAX_COMPRESSED_BYTES` | `50 MiB` | `from-ir` assets zip |
| `PDF2PPT_ZIP_MAX_UNCOMPRESSED_BYTES` | `200 MiB` | … |
| `PDF2PPT_ZIP_MAX_ENTRIES` | `10000` | … |
| `PDF2PPT_ZIP_MAX_ENTRY_BYTES` | `50 MiB` | … |
| `PDF2PPT_ZIP_MAX_COMPRESSION_RATIO` | `100` | … |
| `PDF2PPT_LAYOUT_PLANNER_MODEL` | `gemini-3.1-flash-lite` | Planner model override |
| `PDF2PPT_LAYOUT_PLANNER_INPUT_USD_PER_MTOK` / `_OUTPUT_…` | `0.25` / `1.50` | Cost accounting |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | unset | Planner / generative Gemini |
| `OPENAI_API_KEY` | unset | Generative OpenAI |
| `PDF2PPT_GEMINI_IMAGE_MODEL` / `PDF2PPT_OPENAI_IMAGE_MODEL` | product defaults | Generative image models |
| `PDF2PPT_API_PORT` / `PDF2PPT_UI_PORT` | `8008` / `8080` | Compose host ports |
| `PDF2PPT_MODEL_DIR` | named volume `pdf2ppt-models` | Compose model mount |

Never commit real keys. Template: [`.env.example`](.env.example).

**Layout planner brakes (CLI defaults):** max calls **30**, max cost **$1.0**,
timeout **45s**, concurrency **2**, circuit after **2** consecutive failures
(auth fails open the circuit immediately). Details: [`docs/layout-planner.md`](docs/layout-planner.md).

## Development / testing

```bash
python -m pip install -e ".[dev]"
make lint          # ruff + mypy
make test          # pytest
make test-frontend # frontend unit tests
make help          # all shortcuts
```

CI (`.github/workflows/ci.yml`): pytest, ruff + mypy, frontend tests/build,
wheel smoke test, report-only slide-quality bench.

## Known limitations

- **Rotated pages (90/180/270°):** SlideIR / overlay element boxes use unrotated
  page coordinates and may not match the preview image (same as CLI PPTX). Prefer
  raster fallback on those pages.
- **Wide non-wrapping text:** LibreOffice often ignores `wrap=none` on
  placeholders; title widening and OCR single-line heuristics mitigate but do not
  eliminate reflow vs PowerPoint.
- **OCR timeout lane:** After a page timeout the OCR worker thread is replaced;
  Paddle is thread-affine, so ~30–40% of replacements can raise native
  `std::exception`. The pipeline degrades to empty OCR / raster fallback for that
  page instead of failing the whole job. Prefer the default 120s timeout.
- **Refine time reserve:** The refine loop refuses to start a trial when remaining
  budget is below the issued soffice reserve (`max(5s, 3×` last single-page
  render)), so a tiny leftover crumb cannot kill a healthy convert. Default deck
  budget is **90s**; counts are machine-dependent.
- **Batch SIGINT during native OCR:** Summary files stay valid; the process may
  still exit with a non-zero abnormal code — rerun with `--resume`.

## Troubleshooting / FAQ

| Issue | Fix |
| --- | --- |
| `Failed to import PaddleOCR` / missing OCR extra | `pip install 'pdf2ppt[ocr]'`; keep `numpy<2` |
| OCR breaks after upgrading Paddle | Stay on **paddlepaddle==3.2.0** (3.3.x breaks PP-OCRv5 on CPU) |
| `--refine` skipped | Install LibreOffice Impress (`soffice` on PATH); Docker: `INSTALL_LIBREOFFICE=1` |
| Page / OCR timeouts, empty OCR after hang | Raise `--page-timeout-s` / `PDF2PPT_PAGE_TIMEOUT_S`; see OCR lane limitation above |
| `409 job-busy` | Another request holds the job; UI retries preview-edits (1/2/4s). Wait or retry Convert manually |
| `413` on preview-edits | Body exceeds **1 MiB** — reduce edit payload |
| Docker permission / model dir not writable | Named volume `pdf2ppt-models`, or `chown 10001:10001` host bind; CLI: `docker run --user "$(id -u):$(id -g)"` |
| `PDF2PPT_CORS_ORIGINS=*` | Rejected while credentials are enabled; list explicit origins |
| fastapi / pydantic conflicts | Clean venv + `[server]` |
| CJK / refine metrics look wrong | Install `fonts-noto-cjk` |

## Docs

| Doc | Contents |
| --- | --- |
| [`docs/README.md`](docs/README.md) | **Index** of all docs |
| [`docs/quickstart.md`](docs/quickstart.md) | Install, first convert, Docker |
| [`docs/cli-reference.md`](docs/cli-reference.md) | Flags, batch, exits, OCR timeout |
| [`docs/review-ui.md`](docs/review-ui.md) | Review UI + API + preview |
| [`docs/slideir.md`](docs/slideir.md) | SlideIR emit-ir / from-ir |
| [`docs/layout-planner.md`](docs/layout-planner.md) | Gemini planner caps / circuit |
| [`docs/layout-analysis.md`](docs/layout-analysis.md) | Heuristic / PP-Structure |
| [`docs/vector-nativization.md`](docs/vector-nativization.md) | Shapes + borderless/zebra tables |
| [`docs/background-engines.md`](docs/background-engines.md) | Inpaint engines / `auto` |
| [`docs/generative-inpaint.md`](docs/generative-inpaint.md) | Cloud generative backends |
| [`docs/slide-quality.md`](docs/slide-quality.md) | Fidelity / refine evaluation |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

## License

MIT. See [`LICENSE`](LICENSE).
