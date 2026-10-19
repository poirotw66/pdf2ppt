Checked-in test fixtures belong here.

Use this directory for small, deterministic input files that are required by automated tests.
Do not write generated PPTX, JSON reports, or debug artifacts here.

Consumers:

- `scripts/bench_inpaint.py` — inpainting-engine metrics
- `scripts/bench_slide.py` — end-to-end slide fidelity / editability (see `docs/slide-quality.md`)
- Regenerate with `python tests/fixtures/generate_fixtures.py`