# Background reconstruction engines

`pdf2ppt` rebuilds slide backgrounds only when needed. Overlay pages erase
text that will become editable PowerPoint boxes, then fill the hole with one
of the engines below.

## Engines at a glance

| Engine | When to use | GPU |
| --- | --- | --- |
| `opencv-fast` | Recommended default for most slides | No |
| `white-box` | Safe fallback for large / risky masks | No |
| `lama-onnx` | Optional higher-quality local repair (CPU or GPU) | Optional |
| `lama-pytorch` | Optional repair via official LaMa PyTorch checkpoint | Recommended |
| `generative-openai` / `generative-gemini` | Cloud image-edit backends | N/A |
| `auto` | Route by mask size and available runtimes | Depends |

See also [`generative-inpaint.md`](generative-inpaint.md) for the cloud engines.

`diffusion-local` was removed: local diffusion inpainting was too slow and
inconsistent for this pipeline.

## How `opencv-fast` works

`opencv-fast` is the lightweight local background reconstruction path used for
overlay pages. It is designed for speed and low setup cost:

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
- Image path: RGB PIL image → BGR OpenCV array → surface-fit prefill for smooth components → Telea fallback for residual mask → boundary blend → RGB PIL image

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

## Optional engine: `lama-onnx`

`lama-onnx` is an explicit opt-in engine for overlay background reconstruction
(`lama-onnx-cuda` is a backward-compatible alias; `lama-onnx-cuda-hybrid` /
`lama-onnx-hybrid` are the hybrid-mode equivalents). It does not require a GPU:

- Install the optional runtime with `pip install .[cpu]` (CPU-only `onnxruntime`) or `pip install .[gpu]` (`onnxruntime-gpu`, for CUDA)
- Place a local ONNX model under `model/lama/` or point `--inpaint-model-root` at a specific `.onnx` file
- `--inpaint-onnx-cuda-provider` (default `CUDAExecutionProvider`) is used when available; otherwise the engine falls back to `CPUExecutionProvider`
- When `--inpaint-engine lama-onnx` is selected, a missing runtime or model is a hard error; it does not silently fall back to `opencv-fast`
- Large overlay images are proportionally downscaled to `--inpaint-max-side-px` before inference

`auto` routing also uses `lama-onnx` instead of `white-box` for large masks when
a LaMa runtime is detected (an importable `onnxruntime` package and an existing
`--inpaint-model-root`). When no runtime is detected — the default for most
installs, since `onnxruntime` is an optional extra — `auto` keeps the previous
`white-box` fallback.

## Optional engine: `lama-pytorch`

`lama-pytorch` is an explicit opt-in engine that runs the official LaMa PyTorch
checkpoint through a separate Python environment. Unlike `lama-onnx`, it is
never selected by `auto` routing (the large-mask fallback only considers
`lama-onnx`).

What you need:

- Clone the official LaMa repository into `./lama`
- Extract the `big-lama` checkpoint under `./lama/big-lama` with `config.yaml` and `models/best.ckpt`
- Create a dedicated Conda environment for LaMa inference (separate from the `ppocr` runtime)

Recommended LaMa environment setup:

```bash
bash scripts/setup_lama_env.sh
conda activate lama
```

The setup script creates a `lama` Conda env with PyTorch CUDA, pins `numpy<2`,
and installs the LaMa repo dependencies. Override the env name with
`LAMA_CONDA_ENV=my-lama-env`.

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

- By default, pdf2ppt looks for the `lama` Conda env's Python (or `PDF2PPT_LAMA_PYTHON`)
- Override explicitly with `--inpaint-lama-python /path/to/python`
- The `ppocr` environment runs pdf2ppt; the `lama` environment only runs LaMa inference

Performance characteristics:

- pdf2ppt starts a persistent LaMa worker (`lama/bin/pdf2ppt_predict_server.py`) that loads the checkpoint once per conversion process
- The first overlay page pays the model-load cost (often around 10 seconds on GPU)
- Later pages reuse the loaded model and are much faster (often around 1 to 2 seconds)
- Large pages are proportionally downscaled to `--inpaint-max-side-px` before inference, then resized back to the original page size

Important notes:

- `lama-pytorch` expects the extracted PyTorch checkpoint directory (`lama/big-lama`)
- `lama-onnx` expects an `.onnx` file under `model/lama/`
- When `--inpaint-engine lama-pytorch` is selected, missing repo, checkpoint, or LaMa runtime dependencies is a hard error
- If you need the fastest GPU path and already have a local ONNX export, prefer `lama-onnx`

## Background modes and `auto` routing

The project does not always remove text from the entire page image. It uses a
conditional strategy:

- `elements`: keep editable/native elements without creating a full background image
- `overlay`: rebuild the background behind reconstructed text only
- `full-page`: fall back to a full-page image when editability is too risky

For overlay pages, `auto` routing chooses:

1. Measure the text-mask area ratio.
2. If it is at or below `--inpaint-max-area-ratio`, use `opencv-fast`.
3. If it is above that threshold, estimate how much of the masked region is low-texture.
4. Keep `opencv-fast` when that large-mask exception still looks safe (capped internally so very large masks skip this exception).
5. Otherwise, use `lama-onnx` if a LaMa runtime is detected; use `white-box` if not.

Background complexity is still measured and written into debug notes, but it is
no longer the primary branch point.

## Relevant knobs

- `--inpaint-engine`: `auto`, `white-box`, `opencv-fast`, `lama-onnx` (alias `lama-onnx-cuda`), `lama-pytorch`, `generative-openai`, or `generative-gemini`
- `--inpaint-padding-px`: enlarge the text mask before inpainting
- `--inpaint-max-area-ratio`: avoid using local repair when too much of the page is masked
- `--inpaint-model-root`: checkpoint directory for `lama-pytorch`, or directory / `.onnx` file for `lama-onnx`
- `--inpaint-lama-repo-root`: official LaMa repository root for `lama-pytorch`, default `./lama`
- `--inpaint-lama-device`: device passed to LaMa PyTorch inference, default `cuda`
- `--inpaint-lama-python`: Python executable used for `lama-pytorch`
- `--inpaint-onnx-cuda-provider`: ONNX Runtime provider name for `lama-onnx`
- `--inpaint-onnx-execution-mode`: `sequential` or `parallel` ONNX Runtime execution mode
- `--inpaint-max-side-px`: maximum image side sent into LaMa engines before proportional downscaling
- `--debug-dir`: inspect generated masks and background decisions

Practical guidance:

- Start with `opencv-fast` for most presentation PDFs.
- Increase `--inpaint-padding-px` slightly if text halos remain.
- If `opencv-fast` cannot safely repair a large mask, prefer `white-box` over a slower generative fallback unless you have intentionally set up `lama-onnx` or `lama-pytorch`.
