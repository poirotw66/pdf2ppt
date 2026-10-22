Checked-in test fixtures belong here.

Use this directory for small, deterministic input files that are required by automated tests.
Do not write generated PPTX, JSON reports, or debug artifacts here.

Consumers:

- `scripts/bench_inpaint.py` — inpainting-engine metrics
- `scripts/bench_slide.py` — end-to-end slide fidelity / editability (see `docs/slide-quality.md`)
- `scripts/bench_layout_planner.py` — Phase-3 layout planner acceptance metrics (see `docs/layout-planner.md`)
- Regenerate with `python tests/fixtures/generate_fixtures.py`

P1 vector/table fixtures (in addition to the Phase 0 inpaint set):

- `vector_shapes.pdf` — rects, lines, ellipse, rounded box, arrow
- `flowchart.pdf` — flowchart boxes and connectors
- `merged_table.pdf` — bordered table with a merged header cell
- `geometric_chart.pdf` / `table.pdf` — also used as vector-nativization targets
- `borderless_table.pdf` — aligned text columns with no stroked rules
- `zebra_borderless_table.pdf` — alternating row fills without gridlines
- `table_over_photo.pdf` — transparent ruled table over a full-bleed photo
- `form_tall_empty_cell.pdf` — form with a large empty cell / top-aligned neighbour

P2 layout fixtures (plus optional `*.layout.json` ground truth for grouping metrics):

- `bullet_list.pdf` — multi-level bullets, title, footer/page number
- `title_two_column.pdf` — title + two body columns
- `figure_caption.pdf` — embedded figure with caption under it
- `scanned_bullet_list.pdf` — rasterized bullet slide (OCR / scanned path)
- `flowchart.pdf` — also used to assert side-by-side labels do not merge

Phase 3 layout-planner annotations (`*.planner.json`) plus recorded VLM JSON under
`layout_planner_recorded/` for CI-safe `--mode recorded` evaluation.