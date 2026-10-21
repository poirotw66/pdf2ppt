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
- Vector / table nativization for simple drawings and ruled tables on digital PDFs ([`docs/vector-nativization.md`](docs/vector-nativization.md))
- SlideIR agent layer: emit a versioned JSON IR (`emit-ir`) and render PPTX from IR alone (`from-ir`) so agents can round-trip without re-running the PDF pipeline ([`docs/slideir.md`](docs/slideir.md))
- Background engines: `opencv-fast` (recommended), `white-box`, optional `lama-onnx` / `lama-pytorch`, and cloud generative backends ([`docs/background-engines.md`](docs/background-engines.md))
- Optional deterministic render-compare-refine loop (`--refine`, experimental) after PPTX build: budget-aware baseline (full or partial), continuous severity ordering, leftover-time reallocation, per-element attempt reset after accepts
- JSON conversion report, debug artifacts, and an optional OCR review UI

## Install

Requirements: Python 3.11+, Linux recommended. For OCR, use a dedicated
environment with `numpy<2` (PaddleOCR / PaddleX are more reliable that way).

```bash
cd pdf2ppt
conda create -n ppocr python=3.12 numpy=1.26.4 -y
conda activate ppocr
# Or with venv: python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
```

If you already have an environment:

```bash
python -m pip install "numpy<2"
python -m pip install -e .
```

Always use the editable install from this repo (`pip install -e .`). Do not
point `PYTHONPATH` at `build/lib/` — that directory is a stale setuptools
artifact and is gitignored.

Optional: `python -m pip install pytest` for tests. OCR models default to
`./model` (override with `--ocr-model-root`).

## Quick start

```bash
pdf2ppt input.pdf output.pptx
pdf2ppt input.pdf output.pptx --report output.report.json --debug-dir output_debug
pdf2ppt input.pdf output.pptx --inpaint-engine opencv-fast

# Experimental opt-in P4 render-compare-refine (LibreOffice required; skipped with a warning if missing)
pdf2ppt input.pdf output.pptx --refine --refine-iters 2

# Agent / SlideIR round-trip (no PPTX until from-ir)
pdf2ppt emit-ir input.pdf deck.slideir.json
pdf2ppt from-ir deck.slideir.json rebuilt.pptx
```

### Review UI (optional)

```bash
# Terminal 1 — API
cd pdf2ppt
conda activate ppocr   # or: source .venv/bin/activate
python -m uvicorn pdf2ppt.api:app --host 127.0.0.1 --port 8008

# Terminal 2 — frontend
cd pdf2ppt/frontend
npm install            # first time / after dependency changes
npm run dev
```

Open `http://127.0.0.1:5173`. Full workflow, job storage, and API details:
[`docs/review-ui.md`](docs/review-ui.md).

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
| [`docs/cli-reference.md`](docs/cli-reference.md) | Flags, modes, outputs, limitations |
| [`docs/background-engines.md`](docs/background-engines.md) | Inpaint engines and `auto` routing |
| [`docs/review-ui.md`](docs/review-ui.md) | Review UI setup and API |
| [`docs/layout-analysis.md`](docs/layout-analysis.md) | Heuristic / PP-Structure layout |
| [`docs/vector-nativization.md`](docs/vector-nativization.md) | Shape and table nativization |
| [`docs/slideir.md`](docs/slideir.md) | SlideIR emit-ir / from-ir agent layer |
| [`docs/generative-inpaint.md`](docs/generative-inpaint.md) | Cloud generative backends |
| [`docs/slide-quality.md`](docs/slide-quality.md) | Fidelity / editability evaluation |

CI (`.github/workflows/ci.yml`) runs pytest, ruff + mypy, frontend tests and
build, plus a report-only slide-quality bench.

## License

MIT. See [`LICENSE`](LICENSE).
