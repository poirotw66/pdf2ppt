# Vector and table nativization (P1)

Digital PDFs often store rectangles, lines, ellipses, and ruled tables as vector
drawings. Before P1, pdf2ppt rasterized those into a full-page background image
and only kept text boxes editable. P1 converts simple vectors and detected
tables into **native PowerPoint objects** so slides stay editable without baking
shapes into the background.

## Behaviour

| Page kind | Default (`--vector-mode auto`) |
|-----------|--------------------------------|
| `digital` / `hybrid` | Extract drawings via PyMuPDF `page.get_drawings()` and tables via `page.find_tables()`. Map simple paths to autoshapes / connectors / freeforms; emit native tables; erase nativized regions from any remaining raster background. |
| `scanned` | Unchanged (OCR + overlay / full-page). |

### Shape mapping

| PDF path | PowerPoint object |
|----------|-------------------|
| Single `re` rectangle | `RECTANGLE` (page-covering fills become page-background shapes) |
| Four cubic arcs ≈ ellipse | `OVAL` |
| Lines + quarter-arcs ≈ rounded box | `ROUNDED_RECTANGLE` |
| Single line | Straight connector |
| Closed polyline / simple polygon | Freeform or `RIGHT_ARROW` when elongated |
| Dense / gradient / clipped / huge path counts | Left in the raster background |

Style preserved when available: fill colour, stroke colour, stroke width, dash,
fill/stroke opacity, and drawing order (`seqno` → z-order).

### Tables

- Detected with `find_tables()` (prefers `refine=True` when available).
- **Borderless / zebra fallback:** when ruled detection misses a grid,
  `table_detection.detect_inferred_tables` recovers column-aligned text
  (stable x-columns + row baselines) and alternating filled row rectangles as
  native tables. Guards reject multi-column body copy, bullet lists, and card
  layouts (see `tests/test_table_reconstruction.py`).
- Emits a native `pptx` table with cell text, column widths / row heights,
  cell fills, borders, and merges (`colspan` / `rowspan`) when PyMuPDF reports
  them. Cell fills prefer an **opaque** containing filled vector rect, then the
  modal non-text pixel colour (≥60% share, 4-bit quantised), then the median of
  non-text pixels (gradients / soft fills). Translucent vector fills skip the
  vector step so composited colours are sampled from the rendered page.
  When a table sits on an embedded photo and a cell has **no** source fill, the
  cell stays transparent (no raster sampling) and the table region is **not**
  erased from the background so the photo remains visible under the grid.
  Absorbed cell word boxes are still added to the overlay inpaint mask
  (1–2pt pad) so baked-in glyphs do not ghost under editable PPTX text.
  Inferred column edges use the gap between adjacent columns' text extents
  (not midpoints of left edges); horizontal alignment is majority-voted per
  column. Cell margins are measured from source insets when available;
  otherwise PowerPoint defaults are kept (margins are never zeroed globally).
  Candidates with ≤2 cells covering ≥80% of the page, or grids with neither
  stroked rules, hairline filled-rect rules, nor zebra row fills, are rejected.
  Bullet / numbered list columns and grids with no short/numeric column are
  rejected so Strengths/Risks-style two-column lists stay text boxes.
- Tall / largely empty form cells top-anchor text when the first line sits near
  the top of the cell; tall cells with balanced gaps (merged headers) stay
  vertically centred.
- Grid lines and fills that sit inside the table bbox are **not** also emitted
  as shapes.
- Text blocks whose content belongs to the table are **not** also emitted as
  text boxes.

### Background

- If every drawing was nativized and there are no embedded images → `elements`
  mode (no full-page raster).
- Otherwise → `overlay` / `full-page` as today, but nativized shape/table regions
  are solid-filled out of the background so they are not duplicated. Each erase
  uses the median colour of a small ring around the region (falling back to an
  underneath opaque fill) so white halos do not appear when a large card sits
  under the page raster.
- Opaque fills painted **before** an overlapping picture are marked
  `render_under_raster` so PPTX places them under the background image (paint
  order alone when `get_bboxlog()` is available; size/overlap fallback when it
  is not). Badges / cards painted after a photo stay editable above. This is
  distinct from `is_page_background` (≥95% page fills).
- Translucent solid fills are **un-composited** from the raster
  (`bg = (pix − α·fill) / (1 − α)`) before the native shape is drawn, so
  caption boxes over photos are not darkened twice.
- `--vector-mode off` restores the pre-P1 path exactly for regression.

## CLI / API

```bash
pdf2ppt input.pdf output.pptx --vector-mode auto
pdf2ppt input.pdf output.pptx --vector-mode off
pdf2ppt input.pdf output.pptx --vector-mode aggressive --vector-complexity-threshold 120
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--vector-mode {off,auto,aggressive}` | `auto` | Master switch |
| `--vector-complexity-threshold` | `80` | Max drawings in `auto`. `aggressive` uses `max(threshold, 400)`. |

`ConversionOptions.vector_mode` and `vector_complexity_threshold` mirror the CLI.

## Evaluator

`render_compare.inspect_slide_editability` counts auto-shapes, lines, tables,
and **non-empty table cells** (as text units alongside text boxes). The
composite score uses text 0.40 / structure 0.45 / base 0.15 so a fully native
table page is not capped near 0.50 just because cell text left the text-box
count.

Fixtures added for P1:

- `vector_shapes.pdf` — rects, lines, ellipse, rounded box, arrow
- `flowchart.pdf` — boxes + connectors
- `geometric_chart.pdf` — bar chart of vector rects (existing, now nativized)
- `merged_table.pdf` — bordered table with a merged header
- `table.pdf` — ruled grid (existing)

```bash
python scripts/bench_slide.py --ocr off \
  --fixture geometric_chart --fixture table \
  --fixture vector_shapes --fixture flowchart --fixture merged_table
```

## Limits

- Gradients, soft masks, patterns, and very dense illustrations stay raster.
- Merge detection depends on PyMuPDF’s table finder; inferred borderless tables
  do not invent merges.
- LibreOffice ≠ PowerPoint metrics still apply when scoring fidelity
  (`docs/slide-quality.md`).
