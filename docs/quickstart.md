# Quick start

Short path from zero to a PPTX. For flags and architecture see the root
[README](../README.md) and [cli-reference.md](cli-reference.md).

## Requirements

- Python 3.11+ (3.12 recommended)
- Linux recommended for OCR
- Optional: Node 20+ for the Review UI frontend
- Optional: LibreOffice (`soffice`) only for `--refine` / slide-quality benches
- Optional: `GEMINI_API_KEY` only for `--layout-planner gemini` (never required for default convert)

## Install (pip)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install ".[ocr]"
```

From a published wheel / PyPI-style name once packaged:

```bash
python -m pip install "pdf2ppt[ocr]"
```

Extras:

| Extra | Provides |
| --- | --- |
| *(none)* | Digital/native PDF → PPTX core |
| `[ocr]` | PaddleOCR + **paddlepaddle==3.2.0** (do not use 3.3.1) |
| `[server]` | FastAPI Review UI backend |
| `[dev]` | OCR + server + ruff + mypy + pytest |
| `[all]` | OCR + server + CPU ONNX Runtime |
| `[cpu]` / `[gpu]` | ONNX Runtime for `lama-onnx` |
| `[layout]` | Same Paddle stack as `[ocr]` (PP-Structure discoverability) |

Editable checkout for development:

```bash
python -m pip install -e ".[dev]"
# or: make install
```

### pipx

```bash
pipx install ".[ocr]"          # from a local clone
# pipx install "pdf2ppt[ocr]"  # when published
pdf2ppt --help
```

## First conversion (3 commands)

```bash
python -m pip install ".[ocr]"
pdf2ppt tests/fixtures/solid_background.pdf output.pptx
# equivalent: python -m pdf2ppt tests/fixtures/solid_background.pdf output.pptx
```

Digital fixtures convert without downloading OCR models. Scanned pages need
`[ocr]` and models under `./model` (or `--ocr-model-root`).

## Common flags

```bash
pdf2ppt input.pdf output.pptx --report output.report.json --debug-dir output_debug
pdf2ppt input.pdf output.pptx --inpaint-engine opencv-fast
pdf2ppt input.pdf output.pptx --mode editable
pdf2ppt emit-ir input.pdf deck.slideir.json
pdf2ppt from-ir deck.slideir.json rebuilt.pptx
```

Experimental (optional):

```bash
# LibreOffice required; skipped with a warning if soffice is missing
pdf2ppt input.pdf output.pptx --refine --refine-iters 2

# Requires GEMINI_API_KEY; falls back per page on error
pdf2ppt input.pdf output.pptx --layout-planner gemini
```

## Review UI

```bash
python -m pip install -e ".[ocr,server]"
pdf2ppt-api
# or: python -m uvicorn pdf2ppt.api:app --host 127.0.0.1 --port 8008

cd frontend && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Details: [review-ui.md](review-ui.md).

### Docker Compose

```bash
docker compose up --build
# UI http://127.0.0.1:8080  API http://127.0.0.1:8008/health
```

Env vars (see `.env.example`): `PDF2PPT_ROOT` (image default `/data` so OCR models
land on `/data/model`), `PDF2PPT_JOB_ROOT`, `PDF2PPT_JOB_RETENTION_HOURS`,
`PDF2PPT_CORS_ORIGINS`, optional `GEMINI_API_KEY` / `OPENAI_API_KEY`. Models default
to the named volume `pdf2ppt-models` (owned by container uid 10001). A host path
needs `PDF2PPT_MODEL_DIR=./model` plus `chown 10001:10001` (a plain `./model`
bind is not writable by the container user on Linux). Job artifacts live on
`pdf2ppt-jobs`; conversion downloads go over HTTP (no `./output` bind mount).

One-shot CLI (match host uid so bind-mount writes succeed):

```bash
docker build -t pdf2ppt .
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -w /work pdf2ppt \
  pdf2ppt tests/fixtures/solid_background.pdf /work/output.pptx
```

## SlideIR

```bash
pdf2ppt emit-ir tests/fixtures/solid_background.pdf /tmp/deck.slideir.json
pdf2ppt from-ir /tmp/deck.slideir.json /tmp/deck.pptx
```

Full schema and agent rules: [slideir.md](slideir.md).

## Batch

```bash
pdf2ppt batch ./pdfs -o ./pptx_out
pdf2ppt batch ./pdfs -o ./pptx_out --recursive --resume --skip-existing
```

Exit codes: **0** all ok/skipped, **1** ≥1 failed, **2** usage. After interrupt,
rerun with `--resume` (`summary.*` stays valid). Details: [cli-reference.md](cli-reference.md).

## Troubleshooting

| Symptom | What to try |
| --- | --- |
| `Failed to import PaddleOCR` | `pip install 'pdf2ppt[ocr]'` and keep `numpy<2` |
| OCR crashes / `NotImplementedError` with paddle 3.3.x | Pin `paddlepaddle==3.2.0` (declared in `[ocr]`) |
| CJK glyphs wrong in refine / quality benches | Install `fonts-noto-cjk`; pin family is Noto Sans CJK |
| `--refine` skipped | Install LibreOffice Impress (`soffice` on PATH); Docker `INSTALL_LIBREOFFICE=1` |
| Page / OCR timeouts or empty OCR after a hang | Raise `--page-timeout-s` / `PDF2PPT_PAGE_TIMEOUT_S` (default 120) |
| Review UI `409 job-busy` | Wait; UI retries preview-edits at 1/2/4s. Retry Convert manually |
| Review UI `413` on preview-edits | Body exceeds 1 MiB — shrink the edit payload |
| Docker model dir not writable | Use named volume `pdf2ppt-models`, or `chown 10001:10001` host bind |
| Review UI API will not start / `pdf2ppt-api` hints for `[server]` | `pip install 'pdf2ppt[server]'` in a clean venv |
| `PDF2PPT_CORS_ORIGINS` of only commas / `*` | Falls back to Vite defaults; `*` is rejected with credentials enabled |
| Stale `build/lib` imports | Use `pip install -e .` — never point `PYTHONPATH` at `build/lib/` |

More FAQs and known limitations: root [README](../README.md). Index: [README.md](README.md).
