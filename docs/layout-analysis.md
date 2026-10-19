# Layout analysis (P2)

P2 adds a **layout-analysis stage** that produces typed regions (title, text,
list, figure, table, header/footer, caption, formula, …) plus a reading order.
Those regions improve how pdf2ppt groups text into PowerPoint text boxes and
which roles those boxes get.

## Engines

| `--layout-engine` | Behaviour |
|-------------------|-----------|
| `heuristic` (**default**) | Geometric grouping of existing native/OCR lines. No model download. Used in CI and on CPU-only paths. |
| `ppstructure` | Opt-in PaddleOCR **LayoutDetection** (PP-DocLayout family — the layout stage of **PP-StructureV3**). Soft-falls back to `heuristic` when the model cannot start. |

PaddleOCR is already a core dependency. The `ppstructure` path needs a working
PaddlePaddle runtime and will download DocLayout weights on first use into
`--layout-model-root` (default `./model`). Keep the default `heuristic` unless
you explicitly want neural layout.

Optional install group (discoverability only — no extra pip packages beyond the
Paddle stack you already use for OCR):

```bash
python -m pip install -e ".[layout]"
```

## What layout changes

1. **Text-box grouping** — lines that share a text/list region become one
   paragraph box. Side-by-side flowchart labels (**Collect Info** vs **Verify**)
   stay separate even when their y-ranges overlap.
2. **Bullets** — list regions emit real PowerPoint bullets / auto-numbers
   (`a:buChar` / `a:buAutoNum`) with hanging indent instead of literal glyphs.
3. **Titles** — title regions map to the slide **title placeholder** when a
   Title Only layout is available; the placeholder is raised above the
   background picture/shape in z-order (PDF boldness preserved, no forced wrap).
4. **Reading order** — region order becomes z/tab order for text boxes
   (column-aware: left column then right when both sides have content).
5. **Figures** — figure/chart regions are cropped into discrete picture
   elements unless they overlap already-nativized P1 shapes/tables (avoids
   double ink on flowcharts/charts).
6. **Tables** — table regions **defer to the P1 native table path**. When no
   native table exists (scanned pages), lines stay as separate text boxes
   instead of one overflowing merge.
7. **Headers / footers / page numbers** — OCR HF skipped by default (stay in
   the raster background). Native digital HF stay visible as text boxes.
   Chart axis labels are not treated as footers (`y0 >= 0.93H` + tiny/page-number,
   and never when a shape/table sits within 40pt above).

Works for **digital PDFs** (PyMuPDF spans + layout regions) and **scanned
pages** (OCR lines + layout regions).

## CLI / API

```bash
# Default — heuristic (CI-safe)
pdf2ppt input.pdf output.pptx

# Opt into PP-StructureV3 layout
pdf2ppt input.pdf output.pptx --layout-engine ppstructure

# Keep headers/footers
pdf2ppt input.pdf output.pptx --layout-engine heuristic --keep-header-footer
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--layout-engine {heuristic,ppstructure}` | `heuristic` | Master switch |
| `--layout-model-root` | `./model` | DocLayout / Structure model storage |
| `--keep-header-footer` | off | Retain header/footer/page-number regions |

`ConversionOptions.layout_engine`, `layout_model_root`, and
`layout_skip_header_footer` mirror the CLI. The HTTP API accepts the same
fields on the convert request.

## Evaluator metrics

`pdf2ppt.layout_metrics` scores grouping quality against optional
`*.layout.json` ground truth next to fixtures:

- predicted text-box count vs ground-truth paragraph count
- mean paragraph IoU, merge/split error counts
- list detection precision / recall / F1
- title detection correctness

PPTX helpers also count real bullet paragraphs and title-placeholder usage.
See `tests/test_layout.py` and `docs/slide-quality.md`.

Fixtures added for P2:

- `bullet_list.pdf` (+ `.layout.json`) — multi-level bullets + footer
- `title_two_column.pdf` — title + two body columns
- `flowchart.pdf` — shape labels that must not merge
- `figure_caption.pdf` — figure with caption underneath
- `scanned_bullet_list.pdf` — rasterized bullet slide (OCR path)

```bash
python scripts/bench_slide.py --ocr off \
  --fixture bullet_list --fixture title_two_column \
  --fixture flowchart --fixture figure_caption \
  --extra-option layout_engine=heuristic
```

Compare with `--extra-option layout_engine=ppstructure` when a layout model is
installed. SSIM must not drop materially vs the heuristic baseline.

## Real-model tests

CI mocks LayoutDetection output — no model download. To exercise a real model
locally:

```bash
PDF2PPT_RUN_LAYOUT_MODEL=1 pytest -q tests/test_layout.py -k real_model
```

## Limits

- Heuristic list/title detection is font-size and bullet-prefix based; complex
  magazine layouts still benefit from `ppstructure`.
- Table *structure* remains a P1 concern (`find_tables()`); layout only avoids
  double-emitting text over native tables.
- Title placeholder mapping depends on the blank presentation theme shipping a
  Title Only layout (standard Office themes do).
