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
- `--layout-planner gemini`: **experimental** opt-in VLM layout planner (Phase 3). Structured JSON only; falls back to heuristics per page on error. Requires `GEMINI_API_KEY`. Not in auto/default. Default apply is low-risk (roles / reading order / HF filter); see [`layout-planner.md`](layout-planner.md)
- `--layout-planner-model` / `--layout-planner-max-calls` / `--layout-planner-max-cost-usd` / `--layout-planner-timeout-s` / `--layout-planner-concurrency` / `--layout-planner-cache-dir` / `--layout-planner-min-confidence` / `--layout-planner-max-consecutive-failures`: cost/latency brakes, cache, and circuit breaker
- `--layout-planner-apply-grouping` / `--layout-planner-apply-fonts`: opt into paragraph merges / VLM font remaps (off by default; can hurt SSIM on tables and digital pages)
- `--emit-ir PATH` / `--ir-only` / `--from-ir PATH` / `-o`: SlideIR export and render ([`slideir.md`](slideir.md)). Subcommands: `pdf2ppt emit-ir …`, `pdf2ppt from-ir …`. Assets live in `{ir_filename}.assets/` next to the IR JSON.
- `--keep-header-footer`: keep header/footer/page-number regions (skipped by default)
- `--refine`: **experimental** opt-in P4 render-compare-refine after PPTX build (LibreOffice → PNG, page SSIM, rule-based SlideIR fixes; strict SSIM improvement + editability/text guards). Skipped with a warning if `soffice` is missing or fails. Default off.
- `--refine-iters`: max **accepted improving** iterations per slide (default `2`)
- `--refine-max-fixes`: max accepted fixes per slide (default `8`)
- `--refine-max-attempts`: max attempts per slide including rejects/no-ops (default `8`; ~97% of attempts are rejected on fixture benches)
- `--refine-time-budget`: wall-clock seconds for the whole refine loop (default `90`, must be `>= 1`). Plan roughly **~3s per page** for a useful pass (counts are machine-dependent). Baseline prefers **one full-deck** soffice render (estimate includes diagnosis ~0.45s/page); on short budgets / large decks it may baseline only the highest-risk pages (`baseline_mode=partial`, skipped pages recorded as `time_budget`) and stops diagnosing when the **issued** trial reserve would be breached. Trials render only the page under test. Pages with diagnosed issues are visited by **descending continuous severity** (`40×(1−SSIM)` plus normalised issue terms), then **ascending page index**. First-pass slots are capped at equal fair share. Trial *start* floor is `max(~2.5s, 1.5× per-page render)`; issued soffice timeout / baseline reserve is `max(5s, 3× last successful single-page render)` (decoupled from the slot). Within each issue prefer `widen_bbox` before `shrink_font` / `raster_fallback` (issues stay severity-ordered); duplicate `(element, action, params)` edits and editability/text-F1 prechecks skip without counting. Unused time is reallocated round-robin (up to two attempts per page when affordable). A single trial timeout continues; two consecutive timeouts abort. Otherwise the page is marked `time_budget` (not a false `render_failed`).
- `--refine-min-font-size`: absolute floor (pt) for refine `shrink_font` (default `8`)
- `--refine-min-font-ratio`: never shrink below this fraction of the pre-refine font size (default `0.95`). Effective floor is `max(min-font-size, source * ratio)`. `widen_bbox` pins the declared font size so fit_text cannot grow glyphs.
- `--refine-report`: path for the per-slide refine JSON report (default `<output>.refine.json`)
- `--lang`: PaddleOCR language code, default `ch`
- `--ocr-model-root`: local PaddleOCR model directory, default `./model`
- `--enable-doc-orientation`: enable PaddleOCR document orientation classification; disabled by default for faster startup and OCR
- `--enable-textline-orientation`: enable PaddleOCR textline orientation classification; disabled by default for faster startup and OCR
- `--ocr-det-thresh`: PaddleOCR text detection threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-det-box-thresh`: PaddleOCR detection box threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-drop-score`: PaddleOCR recognition score threshold, optional, uses the PaddleOCR default when omitted
- `--ocr-return-word-box`: return PaddleOCR word-level boxes; disabled by default so pdf2ppt merges same-line detection boxes after OCR
- `--ocr-batch-size`: number of pages processed together for full-page OCR, default `3`
- `--page-timeout-s`: wall-clock seconds for per-page OCR/layout before abandoning that page (default `120`, env `PDF2PPT_PAGE_TIMEOUT_S`; also honoured by the API/library when the request omits the field; `<=0` disables). OCR runs on one persistent daemon worker thread (Paddle thread-affinity). A timed-out OCR **batch** is retried per-page so only the hung page loses OCR. Hung workers are daemon threads so the process can exit; at exit the process waits up to 15s for an abandoned native `predict` to finish (avoids SIGSEGV/SIGABRT during interpreter teardown). A page/job is treated as OCR-wedged only when a *timed-out* call is still running — a lock held by another job's slow-but-live predict is queue wait, not a wedge. **Known limitation:** after a timeout the lane replaces the worker thread; Paddle predictors are thread-affine, so `predict` on the previous engine can raise `std::exception` on some replacements (~30–40%). The pipeline/API then degrade to empty OCR for that page/job (raster fallback) instead of HTTP 502. Prefer a generous timeout (default 120) so timeouts—and thus thread swaps—stay rare.
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
- Experimental VLM layout planner: `--layout-planner gemini` (opt-in, API key). Details: [`layout-planner.md`](layout-planner.md).
- SlideIR (`emit-ir` / `from-ir`) round-trips P1/P2 structures; Phase-3 `groups` / `plan_metadata` are written and rendered when the layout planner succeeds. Details: [`slideir.md`](slideir.md).
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
