# CLI reference

## Main arguments

- `input_pdf`: input PDF path
- `output_pptx`: output PPTX path
- `--report`: path for the JSON conversion report
- `--mode`: `editable`, `fidelity`, or `fast`
- `--vector-mode`: `off`, `auto` (default), or `aggressive` — digital-PDF shape/table nativization ([`vector-nativization.md`](vector-nativization.md))
- `--vector-complexity-threshold`: max drawings considered in `auto` mode (default `80`)
- `--layout-engine`: `heuristic` (default, CI-safe) or `ppstructure` (opt-in PP-StructureV3 / LayoutDetection; soft-falls back to heuristic). See [`layout-analysis.md`](layout-analysis.md)
- `--layout-model-root`: DocLayout / Structure model directory, default `./model`
- `--emit-ir PATH` / `--ir-only` / `--from-ir PATH` / `-o`: SlideIR export and render ([`slideir.md`](slideir.md)). Subcommands: `pdf2ppt emit-ir …`, `pdf2ppt from-ir …`. Assets live in `{ir_filename}.assets/` next to the IR JSON.
- `--keep-header-footer`: keep header/footer/page-number regions (skipped by default)
- `--lang`: PaddleOCR language code, default `ch`
- `--ocr-model-root`: local PaddleOCR model directory, default `./model`
- `--enable-doc-orientation`: enable PaddleOCR document orientation classification; disabled by default for faster startup and OCR
- `--enable-textline-orientation`: enable PaddleOCR textline orientation classification; disabled by default for faster startup and OCR
- `--ocr-det-thresh`: PaddleOCR text detection threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-det-box-thresh`: PaddleOCR detection box threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-drop-score`: PaddleOCR recognition score threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-return-word-box`: return PaddleOCR word-level boxes; disabled by default so pdf2ppt merges same-line detection boxes after OCR
- `--ocr-batch-size`: number of pages processed together for full-page OCR, default `3`
- `--dpi`: render DPI for OCR-oriented page rasterization
- `--background-dpi`: render DPI for embedded full-page and overlay backgrounds
- `--background-format`: `jpeg` or `png` for embedded backgrounds; `jpeg` is smaller
- `--background-jpeg-quality`: JPEG quality used when `--background-format=jpeg`
- `--debug-dir`: directory for per-page debug images and analysis files
- `--enable-doc-unwarping`: enable PaddleOCR UVDoc unwarping
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, or `ERROR`

Background reconstruction flags are documented in
[`background-engines.md`](background-engines.md).

## OCR line merging

Enabled by default (no extra flag required):

- After PP-OCRv5 detection, pdf2ppt merges adjacent boxes on the same text line using vertical alignment, horizontal gap limits, and column-aware x-overlap rules.
- Applies to the CLI, API `detect`, full-page OCR, and boxes shown in the review UI.
- Multi-column slides stay separated when columns do not horizontally overlap, even if their baselines are close.
- Use `--ocr-return-word-box` only for debugging raw PaddleOCR word boxes; keep it off for normal conversions.

## Conversion modes

- `editable`: best balance between editability and visual similarity
- `fidelity`: more conservative rendering, useful when visual match matters more
- `fast`: lower-cost conversion for quick iteration

## Output files

Typical outputs:

- `output.pptx`: editable presentation
- `output.report.json`: structured conversion report
- `output_debug/`: optional debug artifacts

The JSON report includes page classification, background mode, quality scores,
selected background inpainting engine, and OCR/native text blocks with
estimated styling.

## Output size tuning

- Full-page backgrounds still default to `JPEG` at quality `82` and `110 DPI` to reduce PPTX size.
- Overlay backgrounds are embedded as lossless `PNG` automatically so text-removal edges do not pick up JPEG halos.
- Increase `--background-dpi` when visual fidelity matters more than file size.
- Lower `--background-jpeg-quality` further when the deck is still too large.

## OCR model storage

- By default, `pdf2ppt` uses `./model` as the local PaddleOCR model root.
- If the same model already exists in `~/.paddlex/official_models`, the tool reuses it from `./model` through a local link or copy.
- Override the location with `--ocr-model-root /path/to/model` when needed.

## Notes and limitations

- OCR-heavy pages are still estimations, not perfect semantic reconstruction.
- Simple vector drawings and ruled tables on digital pages become native PPT shapes/tables by default (`--vector-mode auto`). Gradients, patterns, clips, and very dense illustrations still fall back to the raster background; use `--vector-mode off` to disable. Details: [`vector-nativization.md`](vector-nativization.md).
- Layout analysis defaults to `heuristic` so CI/CPU paths need no model download. Opt into PP-StructureV3 with `--layout-engine ppstructure`. Details: [`layout-analysis.md`](layout-analysis.md).
- SlideIR (`emit-ir` / `from-ir`) round-trips the same structures P1/P2 already render. `groups` / `plan_metadata` are forward-compatible for P3 and are not rendered today. Details: [`slideir.md`](slideir.md).
- Bold and color restoration for OCR text are heuristic-based.

## Development

```bash
python -m pytest -q
```

### Slide-quality evaluation

To measure whether a change improved visual fidelity or editability, use the
slide-level evaluator. It does not change default conversion behaviour.

```bash
# Metric unit tests (no LibreOffice required)
python -m pytest tests/test_render_compare.py -q

# Full fixture benchmark (requires LibreOffice + preferably fonts-noto-cjk)
sudo apt-get install -y libreoffice-impress fonts-noto-cjk   # Debian/Ubuntu
python scripts/bench_slide.py --ocr off --json-out output/bench_slide.json
```

Fidelity (SSIM, pixel error, edge F1, optional OCR text F1) and editability
(native object counts including shapes/tables, raster-background coverage) are
reported separately so two runs can be diffed in JSON. See
[`slide-quality.md`](slide-quality.md) for metrics, CJK font pinning, and the
LibreOffice vs PowerPoint caveat.

## Project entry points

- CLI: `src/pdf2ppt/cli.py`
- Main pipeline: `src/pdf2ppt/pipeline.py`
- Data models: `src/pdf2ppt/models.py`
