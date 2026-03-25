# pdf2ppt

`pdf2ppt` converts PDF files into editable PowerPoint (`.pptx`) decks.

The project is optimized for presentation-style documents and uses a hybrid pipeline:

- Native PDF extraction first when text and layout can be recovered directly.
- PaddleOCR fallback for scanned or image-heavy pages.
- Page classification (`digital`, `scanned`, `hybrid`) to choose the safest rendering path.
- Conditional background reconstruction for OCR/overlay pages.
- Style recovery for OCR text, including font size fitting, color estimation, and basic bold detection.

## Features

- Convert PDF pages into editable PPTX slides.
- Preserve native text whenever possible.
- Rebuild scanned-page text as editable PowerPoint text boxes.
- Support multiple background reconstruction engines:
  - `white-box`
  - `opencv-fast`
  - `diffusion-local`
  - `auto` routing
- Generate a JSON report for every conversion.
- Generate per-page debug artifacts for OCR masks and background decisions.

## Requirements

- Python 3.11 or newer
- Linux environment recommended
- Dependencies listed in `pyproject.toml`
- For OCR:
  - PaddleOCR runtime and model downloads
- Optional for faster local inpainting:
  - NVIDIA GPU
  - `iopaint` or another compatible local diffusion backend

## Installation

Create and activate your Python environment, then install the project in editable mode:

```bash
python -m pip install -e .
```

If you want to use local diffusion inpainting, install and verify your backend separately. For example, this project has been tested with an `iopaint`-based workflow.

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
- If complexity is high and a diffusion backend is available, `auto` can switch to `diffusion-local`.

Relevant knobs:

- `--inpaint-engine opencv-fast`: force this engine explicitly
- `--inpaint-padding-px`: enlarge the text mask before inpainting
- `--inpaint-max-area-ratio`: avoid using local repair when too much of the page is masked
- `--debug-dir`: inspect generated masks and background decisions

Practical guidance:

- Start with `opencv-fast` for most presentation PDFs.
- Increase `--inpaint-padding-px` slightly if text halos remain.
- Switch to `diffusion-local` only when backgrounds are visually complex enough to justify the extra cost.

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
- `--dpi`: render DPI for OCR and background generation
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
- `diffusion-local` for more complex masked regions, when the backend is available
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
