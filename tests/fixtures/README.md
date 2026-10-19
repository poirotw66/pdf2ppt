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