Checked-in test fixtures belong here.

Use this directory for small, deterministic input files that are required by automated tests.
Do not write generated PPTX, JSON reports, or debug artifacts here.

Consumers:

- `scripts/bench_inpaint.py` — inpainting-engine metrics
- `scripts/bench_slide.py` — end-to-end slide fidelity / editability (see `docs/slide-quality.md`)
- Regenerate with `python tests/fixtures/generate_fixtures.py`

P1 vector/table fixtures (in addition to the Phase 0 inpaint set):

- `vector_shapes.pdf` — rects, lines, ellipse, rounded box, arrow
- `flowchart.pdf` — flowchart boxes and connectors
- `merged_table.pdf` — bordered table with a merged header cell
- `geometric_chart.pdf` / `table.pdf` — also used as vector-nativization targets

P2 layout fixtures (plus optional `*.layout.json` ground truth for grouping metrics):

- `bullet_list.pdf` — multi-level bullets, title, footer/page number
- `title_two_column.pdf` — title + two body columns
- `figure_caption.pdf` — embedded figure with caption under it
- `scanned_bullet_list.pdf` — rasterized bullet slide (OCR / scanned path)
- `flowchart.pdf` — also used to assert side-by-side labels do not merge