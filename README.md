# pdf2ppt

`pdf2ppt` is an open source tool that converts PDF slides into editable PowerPoint (`.pptx`) decks.

It is designed for presentation-style documents and aims to preserve visual fidelity while recovering editable text, layout, and reusable page elements.

Core pipeline:

- Native PDF extraction first when text and layout can be recovered directly.
- PaddleOCR fallback for scanned or image-heavy pages.
- Page classification (`digital`, `scanned`, `hybrid`) to choose the safest rendering path.
- Conditional background reconstruction for OCR/overlay pages.
- Style recovery for OCR text, including font size fitting, color estimation, and basic bold detection.

## Project Status

- Recommended default background engine: `opencv-fast`
- Optional GPU background engines:
  - `lama-onnx-cuda` for explicit high-quality overlay repair when ONNX Runtime CUDA and a local LaMa ONNX model are available
  - `lama-pytorch` for explicit high-quality overlay repair through the official [advimman/lama](https://github.com/advimman/lama) PyTorch checkpoint (`big-lama`)
- `diffusion-local` has been removed because local diffusion inpainting was too slow and inconsistent
- For most documents, start with `opencv-fast` first and let the pipeline fall back to `white-box` when masking is too large

## Result Showcase

<p align="center">
  <img src="image/example_pdf.png" alt="Example source PDF slide" width="48%" />
  <img src="image/example_ppt.png" alt="Example converted PPT slide" width="48%" />
</p>

<p align="center"><em>Source PDF on the left, converted editable PowerPoint on the right.</em></p>

## Features

- Convert PDF pages into editable PPTX slides.
- Preserve native text whenever possible.
- Rebuild scanned-page text as editable PowerPoint text boxes.
- Support multiple background reconstruction engines:
  - `white-box`
  - `opencv-fast` (recommended)
  - `lama-onnx-cuda` (optional GPU path, explicit opt-in)
  - `lama-pytorch` (optional GPU path via official LaMa repo + PyTorch checkpoint, explicit opt-in)
  - `auto` routing
- Generate a JSON report for every conversion.
- Generate per-page debug artifacts for OCR masks and background decisions.
- Show page-level conversion progress in the CLI.

## Requirements

- Python 3.11 or newer
- Linux environment recommended
- Dependencies listed in `pyproject.toml`
- For the OCR stack, use a dedicated Conda environment with `numpy<2`
- For OCR:
  - PaddleOCR runtime and model downloads

## Installation

Recommended setup for OCR-enabled runs:

```bash
conda create -n ppocr python=3.12 numpy=1.26.4 -y
conda activate ppocr
python -m pip install -e .
```

Why this environment is recommended:

- `PaddleOCR` / `PaddleX` currently work more reliably with `numpy<2`.
- A dedicated Conda environment helps avoid conflicts with `pyarrow`, `scikit-learn`, and other globally installed packages.

If you already have an environment, make sure it satisfies the NumPy constraint from `pyproject.toml`:

```bash
python -m pip install "numpy<2"
python -m pip install -e .
```

Optional development dependency for running tests:

```bash
python -m pip install pytest
```

Quick environment check:

```bash
python - <<'PY'
import numpy
print(numpy.__version__)
PY
python -m pdf2ppt input.pdf output.pptx
```

OCR model storage:

- By default, `pdf2ppt` now uses `./model` as the local PaddleOCR model root.
- If the same model already exists in `~/.paddlex/official_models`, the tool reuses it from `./model` through a local link or copy.
- Override the location with `--ocr-model-root /path/to/model` when needed.

## Quick Start

Basic conversion:

```bash
pdf2ppt input.pdf output.pptx
```

Write the report to a specific path:

```bash
pdf2ppt input.pdf output.pptx --report output.report.json
```

Enable debug artifacts:

```bash
pdf2ppt input.pdf output.pptx --debug-dir output_debug
```

Use the fast OpenCV inpainting path:

```bash
pdf2ppt input.pdf output.pptx \
  --inpaint-engine opencv-fast \
  --report output.report.json \
  --debug-dir output_debug
```

### How `opencv-fast` works

`opencv-fast` is the lightweight local background reconstruction path used for overlay pages.

### Optional GPU engine: `lama-onnx-cuda`

`lama-onnx-cuda` is an explicit opt-in engine for overlay background reconstruction. Phase 1 keeps it out of `auto`, so the default route remains stable and CPU-safe.

- Install the optional runtime with `pip install .[gpu]`
- Place a local ONNX model under `model/lama/` or point `--inpaint-model-root` at a specific `.onnx` file
- This path requires `CUDAExecutionProvider` from ONNX Runtime GPU
- When `--inpaint-engine lama-onnx-cuda` is selected, missing runtime, missing provider, or missing model is a hard error; it does not silently fall back to `opencv-fast`
- Large overlay images are proportionally downscaled to `--inpaint-max-side-px` before inference to reduce VRAM pressure

### Optional GPU engine: `lama-pytorch`

`lama-pytorch` is an explicit opt-in engine that runs the official LaMa PyTorch checkpoint through a separate Python environment. Like `lama-onnx-cuda`, it is not part of `auto` routing.

What you need:

- Clone the official LaMa repository into `./lama`
- Extract the `big-lama` checkpoint under `./lama/big-lama` with `config.yaml` and `models/best.ckpt`
- Create a dedicated Conda environment for LaMa inference (separate from the `ppocr` runtime)

Recommended LaMa environment setup:

```bash
bash scripts/setup_lama_env.sh
conda activate lama
```

The setup script creates a `lama` Conda env with PyTorch CUDA, pins `numpy<2`, and installs the LaMa repo dependencies. You can override the env name with `LAMA_CONDA_ENV=my-lama-env`.

Typical conversion command:

```bash
conda activate ppocr
pdf2ppt input.pdf output.pptx \
  --inpaint-engine lama-pytorch \
  --inpaint-model-root lama/big-lama \
  --inpaint-lama-repo-root lama \
  --inpaint-lama-device cuda
```

Python interpreter selection:

- By default, pdf2ppt looks for `~/miniconda3/envs/lama/bin/python` (or `PDF2PPT_LAMA_PYTHON`)
- Override explicitly with `--inpaint-lama-python /path/to/python`
- The `ppocr` environment runs pdf2ppt; the `lama` environment only runs LaMa inference

Performance characteristics:

- pdf2ppt starts a persistent LaMa worker (`lama/bin/pdf2ppt_predict_server.py`) that loads the checkpoint once per conversion process
- The first overlay page pays the model-load cost (often around 10 seconds on GPU)
- Later pages reuse the loaded model and are much faster (often around 1 to 2 seconds)
- Large pages are proportionally downscaled to `--inpaint-max-side-px` before inference, then resized back to the original page size

Important notes:

- `lama-pytorch` and `lama-onnx-cuda` use different model layouts:
  - `lama-pytorch` expects the extracted PyTorch checkpoint directory (`lama/big-lama`)
  - `lama-onnx-cuda` expects an `.onnx` file under `model/lama/`
- When `--inpaint-engine lama-pytorch` is selected, missing repo, missing checkpoint, or missing LaMa runtime dependencies is a hard error
- If you need the fastest GPU path and already have a local ONNX export, prefer `lama-onnx-cuda`

It is designed for speed and low setup cost:

- No model download is required.
- No GPU is required.
- It runs fully in-process through OpenCV.
- It is usually the best default choice for slides with flat or moderately textured backgrounds.

Technical flow:

1. The pipeline first detects text blocks that will be rebuilt as editable PowerPoint text.
2. Those text regions are converted into a binary mask by `build_text_mask_image()`.
3. The mask can be expanded with `--inpaint-padding-px` so the erased region covers anti-aliased text edges and OCR box underestimation.
4. `OpenCvFastInpaintingEngine` converts the page image to a NumPy/OpenCV image.
5. For clearly low-texture or smooth-gradient masked components, the engine first fits a local background surface directly from the surrounding ring pixels.
6. Only the remaining masked regions fall back to `cv2.inpaint(..., cv2.INPAINT_TELEA)`, with nearby small residual components grouped together and the Telea radius adjusted by both component size and surrounding edge density.
7. A final local blend pass smooths the reconstructed patch boundaries before the repaired image is used as the PowerPoint background.

Implementation details:

- Inpainting algorithm: hybrid local surface fitting plus OpenCV Telea fallback (`cv2.INPAINT_TELEA`)
- Base Telea radius: `3.0`, then adjusted by residual component size and surrounding edge density
- Input mask: 8-bit single-channel binary mask
- Image path: RGB PIL image -> BGR OpenCV array -> surface-fit prefill for smooth components -> Telea fallback for residual mask -> boundary blend -> RGB PIL image

Why this method is fast:

- It is a classical image-processing algorithm, not a generative model.
- Smooth regions can often be reconstructed directly from nearby context without invoking iterative inpainting on the full mask.
- The Telea fallback still fills missing pixels by propagating nearby color and structure inward from the mask boundary.
- Runtime is dominated by image size and mask size, not by neural inference or model loading.

When it works well:

- Solid-color slide backgrounds
- Mild gradients
- Light texture or simple shapes behind text
- Quick iteration during tuning and debugging

When it works less well:

- Dense illustrations or photographs directly behind text
- Large masked regions
- Complex patterns where the original content cannot be inferred from nearby pixels
- Cases where erased text overlaps important edges, icons, or thin diagram lines

How it interacts with `auto` routing:

- `auto` first looks at text-mask area ratio, not just visual complexity.
- If the mask area ratio stays within `--inpaint-max-area-ratio`, `auto` keeps using `opencv-fast`.
- If the mask is larger than that threshold, `auto` usually falls back to `white-box`.
- There is one exception: a moderately oversized mask can still use `opencv-fast` when most masked pixels sit on low-texture background.
- The low-texture exception is capped internally, so very large masks still go to `white-box`.
- Background complexity is still measured and written into debug notes, but it is no longer the primary branch point.

Relevant knobs:

- `--inpaint-engine opencv-fast`: force this engine explicitly
- `--inpaint-engine lama-onnx-cuda`: force the optional ONNX GPU engine explicitly
- `--inpaint-engine lama-pytorch`: force the optional PyTorch GPU engine explicitly
- `--inpaint-padding-px`: enlarge the text mask before inpainting
- `--inpaint-max-area-ratio`: avoid using local repair when too much of the page is masked
- `--inpaint-model-root`: checkpoint directory for `lama-pytorch`, or directory / `.onnx` file for `lama-onnx-cuda`
- `--inpaint-lama-repo-root`: official LaMa repository root for `lama-pytorch`, default `./lama`
- `--inpaint-lama-device`: device passed to LaMa PyTorch inference, default `cuda`
- `--inpaint-lama-python`: Python executable used for `lama-pytorch`; defaults to `PDF2PPT_LAMA_PYTHON` or the `lama` Conda env when present
- `--inpaint-onnx-cuda-provider`: ONNX Runtime provider name for `lama-onnx-cuda`
- `--inpaint-onnx-execution-mode`: `sequential` or `parallel` ONNX Runtime execution mode
- `--inpaint-max-side-px`: maximum image side sent into LaMa GPU engines before proportional downscaling
- `--debug-dir`: inspect generated masks and background decisions

Practical guidance:

- Start with `opencv-fast` for most presentation PDFs.
- Increase `--inpaint-padding-px` slightly if text halos remain.
- If `opencv-fast` cannot safely repair a large mask, prefer `white-box` over a slower generative fallback.

## CLI Options

Main arguments:

- `input_pdf`: input PDF path
- `output_pptx`: output PPTX path
- `--report`: path for the JSON conversion report
- `--mode`: `editable`, `fidelity`, or `fast`
- `--lang`: PaddleOCR language code, default `ch`
- `--ocr-model-root`: local PaddleOCR model directory, default `./model`
- `--enable-doc-orientation`: enable PaddleOCR document orientation classification; disabled by default for faster startup and OCR
- `--enable-textline-orientation`: enable PaddleOCR textline orientation classification; disabled by default for faster startup and OCR
- `--ocr-det-thresh`: PaddleOCR text detection threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-det-box-thresh`: PaddleOCR detection box threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-drop-score`: PaddleOCR recognition score threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-batch-size`: number of pages processed together for full-page OCR, default `3`
- `--dpi`: render DPI for OCR-oriented page rasterization
- `--background-dpi`: render DPI for embedded full-page and overlay backgrounds
- `--background-format`: `jpeg` or `png` for embedded backgrounds; `jpeg` is smaller
- `--background-jpeg-quality`: JPEG quality used when `--background-format=jpeg`
- `--debug-dir`: directory for per-page debug images and analysis files
- `--enable-doc-unwarping`: enable PaddleOCR UVDoc unwarping

Background reconstruction:

- `--inpaint-engine`: `auto`, `white-box`, `opencv-fast`, `lama-onnx-cuda`, or `lama-pytorch`
- `--inpaint-padding-px`: expand text masks before inpainting
- `--inpaint-max-area-ratio`: force white-box fallback when the masked area is too large
- `--inpaint-model-root`: checkpoint directory for `lama-pytorch`, or local directory / `.onnx` file for `lama-onnx-cuda`
- `--inpaint-lama-repo-root`: official LaMa repository root for `lama-pytorch`
- `--inpaint-lama-device`: device for `lama-pytorch`, default `cuda`
- `--inpaint-lama-python`: Python executable for `lama-pytorch`
- `--inpaint-onnx-cuda-provider`: ONNX Runtime provider name for `lama-onnx-cuda`
- `--inpaint-onnx-execution-mode`: ONNX Runtime execution mode for `lama-onnx-cuda`
- `--inpaint-max-side-px`: proportional resize guard before LaMa GPU inference

Diagnostics:

- `--log-level`: `DEBUG`, `INFO`, `WARNING`, or `ERROR`

Output size tuning:

- Full-page backgrounds still default to `JPEG` at quality `82` and `110 DPI` to reduce PPTX size.
- Overlay backgrounds are now embedded as lossless `PNG` automatically so text-removal edges do not pick up JPEG halos.
- Increase `--background-dpi` when visual fidelity matters more than file size.
- Lower `--background-jpeg-quality` further when the deck is still too large.

## Conversion Modes

- `editable`: best balance between editability and visual similarity
- `fidelity`: more conservative rendering, useful when visual match matters more
- `fast`: lower-cost conversion for quick iteration

## Background Reconstruction Strategy

The project does not always remove text from the entire page image.

Instead, it uses a conditional strategy:

- `elements`: keep editable/native elements without creating a full background image
- `overlay`: rebuild the background behind reconstructed text only
- `full-page`: fall back to a full-page image when editability is too risky

For overlay pages, `auto` routing can choose:

- `opencv-fast` when the mask stays within the configured area threshold
- `opencv-fast` for some moderately oversized masks when the masked region is mostly low-texture
- `white-box` when the mask is too large or too structurally risky for local repair

In practical terms, the `auto` decision order is:

1. Measure the text-mask area ratio.
2. If it is at or below `--inpaint-max-area-ratio`, use `opencv-fast`.
3. If it is above that threshold, estimate how much of the masked region is low-texture.
4. Keep `opencv-fast` only when that large-mask exception still looks safe; otherwise use `white-box`.

## Output Files

Typical outputs:

- `output.pptx`: editable presentation
- `output.report.json`: structured conversion report
- `output_debug/`: optional debug artifacts

The JSON report includes:

- page classification
- background mode
- quality scores
- selected background inpainting engine
- OCR/native text blocks with estimated styling

## Notes and Limitations

- OCR-heavy pages are still estimations, not perfect semantic reconstruction.
- Complex charts and vector graphics are currently preserved more for appearance than semantic editability.
- Bold and color restoration for OCR text are heuristic-based.

## Development

Run tests:

```bash
python -m pytest -q
```

## How to Enable the Review UI

If you have not installed the project into the current Conda environment yet, do this first:

```bash
cd /home/justin/pdf2ppt
source /home/justin/miniconda3/etc/profile.d/conda.sh
conda activate ppocr
python -m pip install -e .
```

Start the backend in one terminal:

```bash
cd /home/justin/pdf2ppt
source /home/justin/miniconda3/etc/profile.d/conda.sh
conda activate ppocr
python -m uvicorn pdf2ppt.api:app --host 127.0.0.1 --port 8008
```

Start the frontend in a second terminal:

```bash
cd /home/justin/pdf2ppt/frontend
npm install
npm run dev
```

Then open `http://127.0.0.1:5173` in the browser.

Notes:

- `npm install` is only required the first time or after frontend dependency changes.
- The `npm install` command must be run inside `frontend/`, not the home directory.
- The frontend proxies `/jobs/*` to the FastAPI backend on port `8008`.
- If the same Conda environment also contains `iopaint` or older `gradio` / `fastapi` stacks, dependency conflicts can break the API startup. In that case, reinstall `pydantic`, `pydantic-core`, `fastapi`, and `starlette`, or use a dedicated environment for `pdf2ppt`.
- If you only want CLI conversion, you do not need to run the review UI.

Run the review backend:

```bash
conda activate ppocr
python -m uvicorn pdf2ppt.api:app --host 127.0.0.1 --port 8008
```

Run the review frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` in the browser. The Vite dev server proxies `/jobs/*` to the FastAPI backend on port `8008`.

## Review Workflow

The project now includes a lightweight review UI for OCR-heavy decks. It is intended for the cases where raw detection boxes need a human pass before inpainting and PPT generation.

End-to-end flow:

1. Upload a PDF in the web UI to create a job.
2. Run OCR detect to generate candidate text boxes and preview URLs.
3. Review each page and adjust the approved boxes before conversion.
4. Save the approved boxes back to the backend.
5. Run conversion so the pipeline uses approved boxes first, fills missing text with per-box recognition when needed, then generates the PPTX and report.

Job storage and cleanup:

- Uploaded PDFs, OCR detection payloads, approved box payloads, and conversion outputs are stored under `.pdf2ppt_jobs/` by default.
- Page previews are no longer persisted as artifacts. `GET /jobs/{job_id}/pages/{page_number}.jpg` renders the preview on demand from the uploaded PDF and returns JPEG.
- Automatic cleanup runs when a new job is created and removes job directories whose `updated_at` is older than the configured retention window.
- Configure the job root with `PDF2PPT_JOB_ROOT=/path/to/jobs`.
- Configure retention with `PDF2PPT_JOB_RETENTION_HOURS=24`. Set it to `0` to disable automatic cleanup.
- Use `DELETE /jobs/{job_id}` to remove a single job and all of its stored artifacts immediately.

Editor capabilities in the current UI:

- Create a new box by dragging on empty canvas.
- Move an existing box by dragging it.
- Resize a selected box from its four corner handles.
- Use zoom, numeric bbox fields, and nudge buttons for fine correction.
- Edit box text, source, and confidence in the inspector.
- Duplicate or delete the selected box.
- Leave text empty for manually added boxes when you want convert-time OCR recognition to populate it.

Relevant backend endpoints:

- `POST /jobs`
- `POST /jobs/{job_id}/detect` with optional `ocr_batch_size` and `confidence_threshold`
- `PUT /jobs/{job_id}/boxes`
- `POST /jobs/{job_id}/convert` with optional `ocr_batch_size`
- `GET /jobs/{job_id}/pages/{page_number}.jpg`
- `GET /jobs/{job_id}/output.pptx`
- `GET /jobs/{job_id}/report.json`
- `DELETE /jobs/{job_id}`

Detect request options:

- `confidence_threshold` filters out OCR boxes below the given confidence before the response is returned.
- Default: `0.75`

Error response model:

- API errors now return structured JSON under `detail`.
- Shape: `{ "detail": { "code": string, "message": string, "page": number | null } }`
- `page` is only present for page-scoped conversion failures.

Common error codes:

- `input-error`: invalid upload, missing approved boxes, malformed approved box payload
- `ocr-initialization-error`: OCR runtime is unavailable or misconfigured
- `ocr-processing-error`: OCR runtime failed while processing the request
- `page-conversion-error`: a page failed during conversion
- `not-found`: unknown job id or missing generated artifact

Typical endpoint status codes:

- `POST /jobs`: `200`, `400`
- `POST /jobs/{job_id}/detect`: `200`, `404`, `502`, `503`
- `PUT /jobs/{job_id}/boxes`: `200`, `400`, `404`
- `POST /jobs/{job_id}/convert`: `200`, `400`, `404`, `500`, `502`, `503`

## CI

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml`.

It runs:

- backend validation with `pytest`
- frontend validation with `npm test`
- frontend production build with `npm run build`

This keeps Python and frontend regressions on the same pull request signal.

Project entry point:

- CLI: `src/pdf2ppt/cli.py`
- Main pipeline: `src/pdf2ppt/pipeline.py`
- Data models: `src/pdf2ppt/models.py`

## Language Versions

- English: `README.md`
- Traditional Chinese: `README_tw.md`

## License

This project is licensed under the MIT License. See `LICENSE`.
