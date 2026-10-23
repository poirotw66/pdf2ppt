# pdf2ppt

[![CI](https://github.com/poirotw66/pdf2ppt/actions/workflows/ci.yml/badge.svg)](https://github.com/poirotw66/pdf2ppt/actions/workflows/ci.yml)

**English** | [繁體中文](README_tw.md)

`pdf2ppt` converts PDF slides into editable PowerPoint (`.pptx`) decks. It
targets presentation-style documents: recover editable text and layout when
possible, fall back to OCR for scanned or image-heavy pages, and rebuild
backgrounds only where text was removed.

## Showcase

<p align="center">
  <img src="image/example_pdf.png" alt="Example source PDF slide" width="48%" />
  <img src="image/example_ppt.png" alt="Example converted PPT slide" width="48%" />
</p>

<p align="center"><em>Source PDF on the left, converted editable PowerPoint on the right.</em></p>

## What it does

- Native PDF extraction first; PaddleOCR when the page is scanned or image-heavy
- Line-box merging so titles stay coherent without collapsing multi-column layouts
- Optional layout analysis for typed regions, reading order, bullets, and title placeholders ([`docs/layout-analysis.md`](docs/layout-analysis.md))
- Experimental opt-in VLM layout planner (`--layout-planner gemini`) for structured roles, paragraph grouping, and font-class hints ([`docs/layout-planner.md`](docs/layout-planner.md))
- Vector / table nativization for simple drawings and ruled tables on digital PDFs ([`docs/vector-nativization.md`](docs/vector-nativization.md))
- SlideIR agent layer: emit a versioned JSON IR (`emit-ir`) and render PPTX from IR alone (`from-ir`) so agents can round-trip without re-running the PDF pipeline ([`docs/slideir.md`](docs/slideir.md))
- Background engines: `opencv-fast` (recommended), `white-box`, optional `lama-onnx` / `lama-pytorch`, and cloud generative backends ([`docs/background-engines.md`](docs/background-engines.md))
- Optional deterministic render-compare-refine loop (`--refine`, experimental) after PPTX build: budget-aware baseline (full or partial), continuous severity ordering, leftover-time reallocation, per-element attempt reset after accepts. Real-soffice page/accept counts are machine-dependent.
- JSON conversion report, debug artifacts, and an optional OCR review UI

## Install

Requirements: Python 3.11+, Linux recommended. Keep `numpy<2` when using OCR
(PaddleOCR / PaddleX are more reliable that way).

Shorter walkthrough: [`docs/quickstart.md`](docs/quickstart.md).

### pip / venv

```bash
cd pdf2ppt
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[ocr]"          # conversion + OCR
# python -m pip install -e ".[dev]"    # editable: OCR + Review UI API + lint/pytest
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
| `[ocr]` | PaddleOCR + **paddlepaddle==3.2.0** (3.3.1 breaks OCR — do not upgrade) |
| `[server]` | FastAPI Review UI backend (`pdf2ppt-api`) |
| `[dev]` | `[ocr]` + `[server]` + ruff + mypy + pytest |
| `[all]` | `[ocr]` + `[server]` + CPU ONNX Runtime |
| `[cpu]` / `[gpu]` | ONNX Runtime for `lama-onnx` |

Always prefer an editable install from this repo for development
(`pip install -e ".[dev]"`). Do not point `PYTHONPATH` at `build/lib/` — that
directory is a stale setuptools artifact and is gitignored. OCR models default
to `./model` (override with `--ocr-model-root`).

### Docker

```bash
docker compose up --build
# Review UI: http://127.0.0.1:8080
# API health: http://127.0.0.1:8008/health
```

OCR models default to the named volume `pdf2ppt-models` (writable by
container uid 10001). To bind a host directory instead, set
`PDF2PPT_MODEL_DIR=./model` and `chown 10001:10001` it — a plain host-owned
`./model` is not writable by the container user on Linux. Job outputs use the
`pdf2ppt-jobs` volume (no `./output` bind).

One-shot CLI (mount inputs/outputs):

```bash
docker build -t pdf2ppt .
# Match the host uid so writes under the bind mount succeed (image USER is 10001).
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -w /work pdf2ppt \
  pdf2ppt tests/fixtures/solid_background.pdf /work/output.pptx
```

`GEMINI_API_KEY` is **optional** (layout planner / generative Gemini only) and
must be passed at runtime (`-e` / Compose `.env`) — it is never baked into
images. See `.env.example`. LibreOffice is omitted by default; rebuild with
`--build-arg INSTALL_LIBREOFFICE=1` for `--refine`.

## Quick start

Three commands to a PPTX (after install):

```bash
python -m pip install ".[ocr]"
pdf2ppt tests/fixtures/solid_background.pdf output.pptx
# same entry point:
python -m pdf2ppt tests/fixtures/solid_background.pdf output.pptx
```

Common flags:

```bash
pdf2ppt input.pdf output.pptx --report output.report.json --debug-dir output_debug
pdf2ppt input.pdf output.pptx --inpaint-engine opencv-fast

# Experimental opt-in P4 render-compare-refine (LibreOffice required; skipped with a warning if missing)
pdf2ppt input.pdf output.pptx --refine --refine-iters 2

# Experimental opt-in Phase-3 VLM layout planner (requires GEMINI_API_KEY; falls back per page)
pdf2ppt input.pdf output.pptx --layout-planner gemini

# Agent / SlideIR round-trip (no PPTX until from-ir)
pdf2ppt emit-ir input.pdf deck.slideir.json
pdf2ppt from-ir deck.slideir.json rebuilt.pptx
```

### Review UI (optional)

```bash
# Terminal 1 — API
python -m pip install -e ".[ocr,server]"
pdf2ppt-api
# or: python -m uvicorn pdf2ppt.api:app --host 127.0.0.1 --port 8008

# Terminal 2 — frontend
cd frontend
npm install            # first time / after dependency changes
npm run dev
```

Open `http://127.0.0.1:5173`. Full workflow, job storage, and API details:
[`docs/review-ui.md`](docs/review-ui.md). Or use `docker compose up --build`
and open `http://127.0.0.1:8080`.

### Troubleshooting

| Issue | Fix |
| --- | --- |
| `Failed to import PaddleOCR` | Install `[ocr]`; keep `numpy<2` |
| OCR fails after upgrading Paddle | Stay on **paddlepaddle==3.2.0** (3.3.1 breaks PP-OCRv5) |
| CJK metrics / refine look wrong | Install `fonts-noto-cjk` |
| `--refine` skipped | Install LibreOffice Impress (`soffice` on PATH) |
| Conflicting fastapi/pydantic | Use a dedicated venv with `[server]` |

### Preview before convert

In the Review UI, switch to **Preview before convert** after uploading a PDF:

1. Click **Build Preview** — runs analysis → SlideIR (layout planner off; no Gemini key needed) and shows each page image with overlays for text blocks (role + reading order), tables/cells, images, and vector shapes.
2. Toggle overlay layers, exclude individual blocks/tables/images/shapes, change a text block’s role (`title` / `body` / `header` / `footer` / `page_number`), or mark a page as **raster fallback**.
3. Click **Convert** — edits are saved, then PPTX is built with `from-ir` so what you previewed is what you download.

HTTP helpers (same backend): `POST /jobs/{id}/preview`, `GET /jobs/{id}/preview`, `PUT /jobs/{id}/preview-edits`, `POST /jobs/{id}/from-ir`. The default CLI convert path is unchanged when this UI flow is not used.

**Limitations**

- **Exclude vs baked backgrounds:** On `overlay` / `full-page` pages, photos and many shapes may already be painted into the background raster. Excluding an IR shape/image drops the native object only — baked pixels may stay visible. Use **raster fallback** when the preview does not change or you need a full-page image. Preview responses set `baked_background: true` on those pages.
- **Re-analysis clears edits:** Element ids are positional (`layout_text_1_N`). Building preview again deletes saved `preview_edits.json` so Convert cannot fail with unknown ids after OCR box changes.
- **Rotated pages (known, pre-existing):** For pages rotated 90/180/270°, SlideIR bboxes may not match the preview image (same mismatch as CLI PPTX on `main`). Prefer raster fallback on those pages until fixed upstream.

## SlideIR

SlideIR is pdf2ppt’s **versioned JSON intermediate representation** (schema
`1.0`). Agents treat it as a deterministic evidence layer: analyze a PDF once
(`emit-ir`), edit structure fields, then render (`from-ir`) without re-running
the PDF pipeline. Schema: `src/pdf2ppt/schemas/slideir.schema.json`. I/O:
`pdf2ppt.ir_io`. Full detail: [`docs/slideir.md`](docs/slideir.md).

### Main fields

| Level | Field | Meaning |
| --- | --- | --- |
| Deck | `schema_version` | Currently `"1.0"`; reject unknown majors |
| Deck | `slide_width_pt` / `slide_height_pt` | Slide size in PDF points |
| Deck | `assets_dir` | Sibling asset directory name (default `{ir_filename}.assets`) |
| Deck | `source` | Optional source path and page count |
| Deck | `plan_metadata` | Optional; written whenever `--layout-planner` is enabled (including when pages fall back) |
| Page | `page_number` / `page_kind` / `background_mode` | Page index, digital/scanned/hybrid, background mode |
| Page | `text_blocks` | Text boxes (`id`, `bbox`, `text`, style, `block_role`, `reading_order`, `paragraphs`, …) |
| Page | `shapes` / `tables` / `images` | Native shapes, tables, image crops (images reference assets by relative path) |
| Page | `layout_regions` | Layout regions |
| Page | `groups` | Optional element groups (`id`, `member_ids`, optional `bbox`/`confidence`); written only with `--layout-planner-apply-grouping` |
| Page | `plan_metadata` | Optional per-page planner metadata (engine, model, accepted groups, …) |
| Page | `background` | Optional background image asset ref |
| Page | `reading_order` | Optional explicit element-id order |

Assets use **relative paths next to the IR file** (not base64):

```text
deck.slideir.json
deck.slideir.json.assets/
  pages/
    001/
      background.png  # optional; .png/.jpg depending on image data
      image_000.png
```

Treat emitted text as ground truth (`locked: true`). Prefer editing
`block_role`, `reading_order`, and renderer-supported styles over rewriting
strings; do not invent text or point assets outside `assets_dir`. With
`--layout-planner gemini`, a successful plan writes `pages[].plan_metadata`
and applies block roles and reading order to `text_blocks` before the IR is
saved. Paragraph merges and `pages[].groups` are written only with
`--layout-planner-apply-grouping`. `from-ir` renders the structures already
present in the IR.

### Export IR from a PDF

```bash
# IR only (no PPTX)
pdf2ppt emit-ir input.pdf deck.slideir.json
# equivalent:
pdf2ppt input.pdf --emit-ir deck.slideir.json --ir-only

# Convert and also write IR
pdf2ppt input.pdf output.pptx --emit-ir deck.slideir.json
```

Analysis flags (`--vector-mode`, `--layout-engine`, OCR/inpaint options, …)
apply to `emit-ir` the same way as `convert`.

### Edit and rebuild PPTX

```bash
# Subcommand
pdf2ppt from-ir deck.slideir.json rebuilt.pptx
# equivalent flag form:
pdf2ppt --from-ir deck.slideir.json -o rebuilt.pptx
```

HTTP (Review UI backend, default `http://127.0.0.1:8008`):

| Endpoint | Behavior |
| --- | --- |
| `POST /jobs/{id}/emit-ir` | Analysis → SlideIR only |
| `POST /jobs/{id}/preview` | Analyze → SlideIR (planner forced off) + overlay preview JSON |
| `GET /jobs/{id}/preview` | Overlay preview from existing SlideIR + saved edits |
| `PUT /jobs/{id}/preview-edits` | Save exclude / role / raster-fallback edits |
| `POST /jobs/{id}/from-ir` | Apply edits → render job SlideIR to PPTX |
| `GET /jobs/{id}/slideir.json` | Download IR JSON |
| `GET /jobs/{id}/slideir-assets.zip` | Download assets directory as a zip |
| `POST /from-ir` | Multipart: required `ir_file` + optional `assets_zip` → PPTX |

Create the job first with `POST /jobs` (multipart file upload);
`POST /jobs/{id}/emit-ir` then takes a JSON body (`{}` for defaults). IR
downloaded from the API names its assets folder `output.slideir.json.assets`,
so CLI `from-ir` users must unzip into a folder of that name
(`POST /from-ir` handles it automatically).

```bash
curl -o rebuilt.pptx \
  -F "ir_file=@deck.slideir.json;type=application/json" \
  -F "assets_zip=@deck.slideir.assets.zip;type=application/zip" \
  http://127.0.0.1:8008/from-ir
```

### Short example

```bash
pdf2ppt emit-ir tests/fixtures/solid_background.pdf /tmp/deck.slideir.json
# Optionally edit /tmp/deck.slideir.json (e.g. text_blocks[].block_role / reading_order)
pdf2ppt from-ir /tmp/deck.slideir.json /tmp/deck.pptx
```

## Architecture

```text
PDF → page classify (digital / scanned / hybrid)
    → native extract and/or OCR (+ line merge, optional layout)
    → optional vector/table nativization
    → conditional background (elements / overlay / full-page)
    → PPTX + JSON report
    → optional --refine (experimental render-compare-fix; LibreOffice)
    → optional SlideIR emit / from-ir render
```

| Piece | Role |
| --- | --- |
| `src/pdf2ppt/cli.py` | CLI entry |
| `src/pdf2ppt/pipeline.py` | Conversion orchestration |
| `src/pdf2ppt/ir_io.py` | SlideIR serialize / deserialize |
| `src/pdf2ppt/api.py` | Review UI backend |
| `frontend/` | OCR box review UI |

Background strategy in short: keep native elements when safe; for overlay
pages, `auto` prefers `opencv-fast`, then `lama-onnx` when a LaMa runtime is
available, otherwise `white-box`. Details:
[`docs/background-engines.md`](docs/background-engines.md).

## More docs

| Doc | Contents |
| --- | --- |
| [`docs/quickstart.md`](docs/quickstart.md) | Install, first convert, Docker, troubleshooting |
| [`docs/cli-reference.md`](docs/cli-reference.md) | Flags, modes, outputs, limitations |
| [`docs/background-engines.md`](docs/background-engines.md) | Inpaint engines and `auto` routing |
| [`docs/review-ui.md`](docs/review-ui.md) | Review UI setup and API |
| [`docs/layout-analysis.md`](docs/layout-analysis.md) | Heuristic / PP-Structure layout |
| [`docs/vector-nativization.md`](docs/vector-nativization.md) | Shape and table nativization |
| [`docs/slideir.md`](docs/slideir.md) | SlideIR emit-ir / from-ir agent layer |
| [`docs/generative-inpaint.md`](docs/generative-inpaint.md) | Cloud generative backends |
| [`docs/slide-quality.md`](docs/slide-quality.md) | Fidelity / editability evaluation |

CI (`.github/workflows/ci.yml`) runs pytest, ruff + mypy, frontend tests and
build, a packaging wheel smoke test, plus a report-only slide-quality bench.
Local shortcuts: `make help`.

## License

MIT. See [`LICENSE`](LICENSE).
