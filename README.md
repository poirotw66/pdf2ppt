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
- `diffusion-local` is still under active development and should be treated as experimental
- For most documents, start with `opencv-fast` first and only try `diffusion-local` for more complex backgrounds

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
  - `diffusion-local` (experimental / in progress)
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
- Optional for faster local inpainting:
  - NVIDIA GPU
  - `iopaint` or another compatible local diffusion backend

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

If you want to use local diffusion inpainting, install and verify your backend separately. For example, this project has been tested with an `iopaint`-based workflow. At this stage, `diffusion-local` is still experimental, so `opencv-fast` remains the recommended first choice.

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
5. The engine calls `cv2.inpaint(..., cv2.INPAINT_TELEA)` with a small radius.
6. The repaired image is used as the background, and editable text boxes are drawn on top in PowerPoint.

Implementation details:

- Inpainting algorithm: OpenCV Telea method (`cv2.INPAINT_TELEA`)
- Default radius: `3.0`
- Input mask: 8-bit single-channel binary mask
- Image path: RGB PIL image -> BGR OpenCV array -> Telea inpaint -> RGB PIL image

Why this method is fast:

- It is a classical image-processing algorithm, not a generative model.
- It fills missing pixels by propagating nearby color and structure inward from the mask boundary.
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

- If mask coverage is too large, `auto` falls back to `white-box` for safety.
- If the local background complexity is low, `auto` prefers `opencv-fast`.
- Complexity is estimated from the area around the mask using grayscale variance and edge density.
- If complexity is high and a diffusion backend is available, `auto` can switch to `diffusion-local`, though that path is still experimental.

Relevant knobs:

- `--inpaint-engine opencv-fast`: force this engine explicitly
- `--inpaint-padding-px`: enlarge the text mask before inpainting
- `--inpaint-max-area-ratio`: avoid using local repair when too much of the page is masked
- `--debug-dir`: inspect generated masks and background decisions

Practical guidance:

- Start with `opencv-fast` for most presentation PDFs.
- Increase `--inpaint-padding-px` slightly if text halos remain.
- Switch to `diffusion-local` only when backgrounds are visually complex enough to justify the extra cost and experimental behavior is acceptable.

Use a local diffusion backend:

```bash
pdf2ppt input.pdf output.pptx \
  --inpaint-engine diffusion-local \
  --diffusion-command iopaint \
  --diffusion-model runwayml/stable-diffusion-inpainting \
  --diffusion-device cuda \
  --report output.report.json
```

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
- `--dpi`: render DPI for OCR-oriented page rasterization
- `--background-dpi`: render DPI for embedded full-page and overlay backgrounds
- `--background-format`: `jpeg` or `png` for embedded backgrounds; `jpeg` is smaller
- `--background-jpeg-quality`: JPEG quality used when `--background-format=jpeg`
- `--debug-dir`: directory for per-page debug images and analysis files
- `--enable-doc-unwarping`: enable PaddleOCR UVDoc unwarping

Background reconstruction:

- `--inpaint-engine`: `auto`, `white-box`, `opencv-fast`, or `diffusion-local`
- `--inpaint-padding-px`: expand text masks before inpainting
- `--inpaint-max-area-ratio`: force white-box fallback when the masked area is too large

Local diffusion settings:

- `--diffusion-command`: CLI command for the backend, default `iopaint`
- `--diffusion-model`: model name passed to the backend
- `--diffusion-device`: `cuda` or `cpu`
- `--diffusion-max-crop-edge`: maximum crop size sent to the backend
- `--diffusion-complexity-threshold`: auto-routing threshold for complex backgrounds
- `--diffusion-timeout-sec`: timeout for each local diffusion backend call

Diagnostics:

- `--log-level`: `DEBUG`, `INFO`, `WARNING`, or `ERROR`

Output size tuning:

- By default, embedded page backgrounds now use `JPEG` at quality `82` and `110 DPI` to reduce PPTX size.
- Increase `--background-dpi` or switch to `--background-format png` when visual fidelity matters more than file size.
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

- `opencv-fast` for simpler backgrounds
- `diffusion-local` for more complex masked regions when the backend is available, though this path is still experimental
- `white-box` as the safest fallback for large masks or unavailable backends

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
- Local diffusion quality depends heavily on backend availability, GPU memory, and model choice.

## Development

Run tests:

```bash
python -m pytest -q
```

Project entry point:

- CLI: `src/pdf2ppt/cli.py`
- Main pipeline: `src/pdf2ppt/pipeline.py`
- Data models: `src/pdf2ppt/models.py`

## Language Versions

- English: `README.md`
- Traditional Chinese: `README_tw.md`

## License

This project is licensed under the MIT License. See `LICENSE`.
