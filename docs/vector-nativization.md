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
- Emits a native `pptx` table with cell text, column widths / row heights,
  header fills (sampled), borders, and merges (`colspan` / `rowspan`) when
  PyMuPDF reports them.
- Grid lines and fills that sit inside the table bbox are **not** also emitted
  as shapes.
- Text blocks whose content belongs to the table are **not** also emitted as
  text boxes.

### Background

- If every drawing was nativized and there are no embedded images → `elements`
  mode (no full-page raster).
- Otherwise → `overlay` / `full-page` as today, but nativized shape/table regions
  are solid-filled out of the background so they are not duplicated.
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
- Merge detection depends on PyMuPDF’s table finder; unruled tables are not
  forced into native tables.
- LibreOffice ≠ PowerPoint metrics still apply when scoring fidelity
  (`docs/slide-quality.md`).
