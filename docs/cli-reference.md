# CLI reference

## Subcommands

| Command | Purpose |
| --- | --- |
| `pdf2ppt INPUT.pdf OUTPUT.pptx [options]` | Single-file convert (default; also `pdf2ppt convert …`) |
| `pdf2ppt emit-ir …` / `pdf2ppt from-ir …` | SlideIR export / render ([`slideir.md`](slideir.md)) |
| `pdf2ppt batch INPUT_DIR -o OUTPUT_DIR [options]` | Folder batch convert (orchestrates the same pipeline) |

Single-file conversion output is unchanged by the batch command: batch only discovers PDFs, calls the existing convert path per file, and writes summaries.

### Batch mode

```bash
pdf2ppt batch ./pdfs -o ./pptx_out
pdf2ppt batch ./pdfs -o ./pptx_out --recursive --skip-existing --resume
pdf2ppt batch ./pdfs -o ./pptx_out --fail-fast --quiet
pdf2ppt batch ./pdfs -o ./pptx_out --jobs 1   # default; keep 1 unless you have ample RAM
```

Behaviour:

- Discovers `*.pdf` / `*.PDF` (case-insensitive). Non-recursive by default; `--recursive` keeps relative subfolders under the output dir and skips any PDFs already under the output directory. Hidden files/dirs (name starts with `.`) are skipped. Order is stable (case-folded relative path, then exact path) so collision suffixes do not depend on filesystem iteration order. Output-name collision detection is case-insensitive (safe on macOS/Windows).
- Writes `<name>.pptx` per PDF (atomically via a `.pptx.partial` temp file + rename). If two inputs collide on the same output name (e.g. `a.pdf` and `a.PDF`), later files get a deterministic `__N` stem suffix (`a__2.pptx`, …).
- Reuses the same conversion option flags as single-file mode (OCR, layout planner, refine, page timeout, inpaint, …) via shared option parsing — not a second copy of the pipeline.
- Failure isolation: a corrupt/encrypted/OCR-init error on one PDF is recorded as `failed` and the batch continues. A **per-page** `--page-timeout-s` does **not** fail the file: that page degrades to empty OCR / raster fallback and the PDF stays `status=ok` (with higher `raster_fallback_pages` / warnings). Exit code **1** if any file failed; **0** if all succeeded or were skipped (including an empty input folder, which logs a warning); **2** for usage/validation errors. Use `--fail-fast` to stop at the first failure. `SIGINT`/`SIGTERM` leave a valid partial `summary.*` (flushed after every file) so `--resume` can continue; KeyboardInterrupt exits **130**.
- `--jobs N` (default **1**, sequential). Parallel mode uses **processes**. **Warning:** PaddleOCR is memory-heavy; `--jobs > 1` has OOM'd 16 GB machines — raise only with ample RAM.
- `--skip-existing` skips only when the output PPTX exists and is a complete valid OOXML zip; truncated/corrupt pptx files are reconverted. `--resume` keeps earlier `ok` rows (including pages/metrics/seconds) when the matching pptx is still valid; if the pptx was deleted or is invalid, that file is reconverted. Two resumes in a row keep the same ok count and pages.
- Progress lines go to **stderr**; `--quiet` suppresses them. At the end, a markdown summary table (with totals) is printed to **stdout** (also written as `summary.md`) unless `--quiet`.
- Quality summary files in the output dir: `summary.csv`, `summary.json`, `summary.md` — rewritten atomically after every file. Per PDF: file, pages, status (`ok` / `failed` / `skipped`), seconds, OCR used / OCR pages, raster-fallback pages, warnings count, planner cost (if used), and a cheap quality indicator (raster-fallback fraction, mean editable ratio / text confidence; no LibreOffice required). Optional `--measure-quality` renders with soffice and records mean SSIM when LibreOffice is available; skipped gracefully when missing.

### Batch exit codes

| Code | Meaning |
| --- | --- |
| `0` | All PDFs succeeded or were skipped |
| `1` | One or more PDFs failed (others may have succeeded) |
| `2` | Usage / validation error (bad args, missing input dir, …) |

Single-file mode additionally uses exit code `3` for OCR initialization failures on a core install without `[ocr]`.

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
- SlideIR (`emit-ir` / `from-ir`) round-trips P1/P2 structures. With `--layout-planner gemini`, successful plans write `pages[].plan_metadata` and apply roles / reading order; deck-level `plan_metadata` is written whenever the planner is enabled (including page fallbacks). Paragraph merges and `pages[].groups` are written **only** with `--layout-planner-apply-grouping`. Details: [`slideir.md`](slideir.md).
- Bold and color restoration for OCR text are heuristic-based.

### Known limitations (honest)

- **Rotated pages (90/180/270°):** element boxes in SlideIR / overlays use unrotated page coordinates and may not align with the preview image (same as CLI PPTX). Prefer raster fallback.
- **Wide non-wrapping text:** LibreOffice often ignores `wrap=none` on placeholders; title widening and OCR single-line heuristics mitigate but do not eliminate reflow vs PowerPoint.
- **OCR timeout lane:** after a timeout the worker thread is replaced; Paddle is thread-affine so ~30–40% of replacements can raise native `std::exception`. The pipeline degrades to empty OCR / raster fallback for that page (not HTTP 502). Prefer the default 120s timeout so swaps stay rare.
- **Refine time reserve:** trials will not start when remaining budget is below the issued soffice reserve (`max(5s, 3×` last single-page render)); default deck budget **90s**.
- **Batch SIGINT during native OCR:** `summary.*` stays valid for `--resume`, but the process may exit with a non-zero abnormal code.

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
- Batch orchestration: `src/pdf2ppt/batch.py`
- Main pipeline: `src/pdf2ppt/pipeline.py`
- Data models: `src/pdf2ppt/models.py`
