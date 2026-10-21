# Slide-quality evaluation (Phase 0 / P0)

This document describes how to measure whether a pdf2ppt change improved
**fidelity** (does the rendered slide look like the source page?) and
**editability** (how much is native / editable vs. a raster background?).

The two scores are always reported separately. A full-page background image can
look nearly identical to the source while remaining hard to edit.

## Why this exists

`docs/optimization-roadmap.md` Phase 0 notes that quality could not be
quantified. Future work (vector nativization, generative inpainting, VLM
planning) needs a stable, diffable JSON baseline. This evaluator is that
baseline; it does **not** change default conversion behaviour.

## Components

| Piece | Role |
|-------|------|
| `src/pdf2ppt/render_compare.py` | Metrics + LibreOffice render path |
| `scripts/bench_slide.py` | Convert fixtures → score → JSON |
| `tests/test_render_compare.py` | Unit tests for metrics (LibreOffice tests skip if missing) |

Related but distinct: `scripts/bench_inpaint.py` scores **inpainting engines**
on mask cleanliness. The slide evaluator scores the **end-to-end PPTX**.

## Metrics

### Fidelity (rendered slide vs. source PDF page)

1. **SSIM** — structural similarity on grayscale (numpy/OpenCV; no scikit-image).
2. **Pixel MAE / MSE** — mean absolute / squared RGB error on a 0–255 scale.
3. **Edge F1** — Canny edge-map precision/recall F1.
4. **Text F1** — bag-of-tokens F1: native PDF text (ground truth) vs. OCR on the
   rendered slide. `null` when OCR is unavailable (`--ocr off` / auto-degrade).
5. **Inpaint-region MAE** — build a text-free ground-truth background by deleting
   the PDF text layer with PyMuPDF redaction (images/graphics kept), extract the
   full-page background picture from the PPTX, and compute MAE **only inside**
   the native text mask (default pad 8px). Also reports **boundary-ring MAE**
   (thin ring just inside the mask) as a seam-visibility proxy. Useful for
   comparing inpaint engines on smooth gradients where full-slide SSIM is
   insensitive. `null` when the PPTX has no background picture or the page has
   no native text boxes.

### Editability (from PPTX structure; no render required)

- Counts: text boxes, pictures, auto-shapes, tables, lines, groups, charts.
- **`raster_background_area_ratio`** — largest picture covering ≥ 85% of the
  slide area (typical full-page overlay / full-page background).
- **`score`** — composite in `[0, 1]`. Text units = text boxes + **non-empty
  native table cells** (so a fully nativized table is not penalized for moving
  cell text out of text boxes). Structure units weight autoshapes, tables×4,
  lines×0.5, groups, charts×2. Weights: text 0.40, structure 0.45, base 0.15,
  minus a raster-background penalty. See also
  [`docs/vector-nativization.md`](vector-nativization.md).

### Layout grouping (P2; no LibreOffice required)

`pdf2ppt.layout_metrics` compares `PageResult` text boxes to optional
`*.layout.json` ground truth:

- box-count ratio vs ground-truth paragraphs
- mean paragraph IoU, merge/split error counts
- list detection precision / recall / F1
- title detection correctness
- PPTX helpers: real bullet paragraph count, title-placeholder presence

See [`docs/layout-analysis.md`](layout-analysis.md) and `tests/test_layout.py`.

## System dependencies (optional)

Conversion itself does not need these. Only evaluation does:

```bash
# Debian / Ubuntu
sudo apt-get install -y libreoffice-impress fonts-noto-cjk
```

- **LibreOffice** (`soffice`) converts PPTX → PDF headlessly; pages are then
  rasterized with PyMuPDF (same stack as the pipeline).
- **Noto Sans CJK TC** — pdf2ppt already writes this family into OOXML for CJK
  text (`text_style.default_font_family`). Pin the same font for rendering so
  metrics are reproducible across machines.

### LibreOffice ≠ PowerPoint

Font metrics, CJK layout, and anti-aliasing differ between LibreOffice and
Microsoft PowerPoint. Treat these scores as a **reproducible CI / laptop
baseline**, not as a WYSIWYG guarantee of what PowerPoint users will see.
Compare runs on the same renderer + font set.

## How to run

```bash
# Metric unit tests (no LibreOffice required)
python -m pytest tests/test_render_compare.py -q

# Full fixture benchmark (needs LibreOffice)
python scripts/bench_slide.py --ocr off
python scripts/bench_slide.py --json-out output/bench_slide.json

# Digital-only fixtures (no PaddleOCR needed during conversion)
python scripts/bench_slide.py --ocr off \
  --fixture solid_background --fixture dark_background_light_text \
  --fixture geometric_chart --fixture table

# Alternate conversion knobs (local + opt-in generative inpainting)
python scripts/bench_slide.py --inpaint-engine white-box
python scripts/bench_slide.py --inpaint-engine lama-onnx --extra-option inpaint_model_root=model/lama
# Requires OPENAI_API_KEY / GEMINI_API_KEY; see docs/generative-inpaint.md
python scripts/bench_slide.py --inpaint-engine generative-openai \
  --extra-option generative_max_calls=5 --extra-option generative_max_cost_usd=1.0
```

Hybrid fixtures (`gradient_*`, `photo_texture`, `large_mask`) still run
`convert_pdf`, which needs a working PaddleOCR + PaddlePaddle stack. The bench
defaults `--convert-ocr-model-root` to `None` (official PaddleX cache) so an
empty local `model/` directory does not break those cases.

JSON layout (abbreviated):

```json
{
  "cases": [
    {
      "name": "solid_background",
      "pages": [
        {
          "page_number": 1,
          "fidelity": {
            "ssim": 0.91,
            "pixel_mae": 4.2,
            "edge_f1": 0.55,
            "text_f1": null,
            "inpaint_region_mae": 0.49,
            "inpaint_boundary_mae": 1.2
          },
          "editability": { "text_box_count": 3, "raster_background_area_ratio": 1.0, "score": 0.31 }
        }
      ]
    }
  ],
  "summary": { "fidelity": { "ssim_mean": 0.88 }, "editability": { "score_mean": 0.29 } }
}
```

Diff two runs by comparing `summary` and per-fixture `pages[*].fidelity` /
`pages[*].editability` in the JSON files.

## CI

The `slide-quality` job in `.github/workflows/ci.yml` is **report-only**
(`continue-on-error: true`). It installs LibreOffice + Noto CJK, runs the
metric tests, and attempts a short fixture bench with `--ocr off`. The printed
summary and uploaded JSON include `inpaint_region_mae` / `inpaint_boundary_mae`
alongside SSIM and full-frame MAE. Failures there never block merge; the
blocking `backend` job still runs the pure metric unit tests (LibreOffice cases
skip automatically).

## Interface stability for future backends

`evaluate_pptx_against_pdf(pptx, pdf, *, dpi, ocr_engine=None)` only needs the
output PPTX and source PDF. `scripts/bench_slide.py --inpaint-engine …` forwards
conversion knobs into `ConversionOptions`. Measure the generative backends with
`--inpaint-engine generative-openai` or `generative-gemini` (plus budget extras);
see [`generative-inpaint.md`](./generative-inpaint.md).

## P4 render-compare-refine (experimental, opt-in)

`pdf2ppt --refine` runs a deterministic loop after the PPTX is written
(**experimental** — default off; refine-off output stays byte-identical):

1. **Baseline render** via LibreOffice → PDF → PNG (same stack as this
   evaluator). Prefer one full-deck convert. Cost estimates include
   per-page diagnosis (~0.45s/page) as well as soffice render. On short
   `--refine-time-budget` / large decks, if that phase would leave too
   little room for trials, only the highest conversion-time risk pages are
   baselined (`baseline_mode=partial`); the diagnosis loop also stops when
   the trial floor would be breached. Others appear in deck
   `pages_skipped` with reason `time_budget`. Subset slides stay
   pixel-identical to a per-page render. Trial renders use a single-slide
   PPTX for the page under test only.
2. Diagnose page SSIM plus localized issues with **render evidence** ranked
   first (ink spill / extra wrap lines, low local SSIM+MAE) — no LLM.
3. Visit pages with issues by **descending continuous severity**
   (`40×(1−SSIM)` plus normalised issue terms, so fidelity dominates), then
   **ascending page index**. Clean pages skip trials.
4. Apply bounded rule-based fixes to SlideIR / `PageResult` (shrink font
   including paragraph sizes, widen/reposition boxes; raster crop only for
   **non-text** regions, with overlying text glyphs erased from the crop).
   `shrink_font` never goes below `max(--refine-min-font-size, source * --refine-min-font-ratio)`
   (default ratio 0.95). `widen_bbox` pins the declared font size so fit_text
   cannot grow glyphs in the larger box.
5. Keep a fix only when page SSIM **strictly improves** and editability / text
   F1 do not regress; XML no-ops are rejected and do **not** consume the attempt
   cap; real attempts are capped per element (default 2) and the per-element
   counter **resets after an accepted fix** (so a widen can be followed by a
   shrink), still bounded by slide/global caps. Caps:
   `--refine-max-attempts` (default 8), `--refine-time-budget` (default 90s;
   plan ~**3s per page**). First-pass slots are capped at the equal fair
   share. A trial may *start* when remaining ≥ `max(~2.5s, 1.5× per-page
   render)`, but the issued soffice timeout is decoupled:
   `max(5s, 3× last successful single-page render)`. Baseline/diagnose
   planning reserves that same issued amount so short budgets (e.g. 10s)
   still leave a real trial slot. Within each issue, prefer `widen_bbox`
   before `shrink_font` / `raster_fallback`; issues themselves stay in
   severity order (not a global widen-first pass). Duplicate
   `(element, action, params)` edits and editability/text-F1 precheck
   failures skip without counting an attempt. Unused time returns to the
   pool; a **round-robin second pass** (up to two attempts per page when
   the budget can afford it) continues while candidates remain. A single
   trial timeout rejects that candidate and continues; two consecutive
   timeouts abort refine. Per-page report fields include `severity`,
   `slot_seconds` / `borrowed_seconds` / `time_used_seconds` (actual used),
   `granted_seconds`, `attempt_count`, and `accepted_fix_count`.
   Wall-clock counts in benches are **machine-dependent**.

When `soffice` is unavailable **or fails** (including mid-loop), the loop
restores the last accepted PPTX, sets deck-level `skipped_reason`, and
conversion still succeeds. See `src/pdf2ppt/refine.py` and `tests/test_refine.py`.
