# Layout planner (experimental, Phase 3 / v1.5)

Opt-in **VLM layout planner** that improves text structure and editability.
It sends each page image plus existing OCR/geometry blocks to a light Gemini
model and receives **structured JSON only** (never pixels). Invalid /
timed-out / low-confidence responses fall back to the existing geometric
heuristics **per page**.

This closes the gap noted in [`spec.md`](./spec.md) §1 (“Paddle Layout Analysis
尚未實作”) via the VLM spike path recommended in
[`optimization-roadmap.md`](./optimization-roadmap.md) Phase 3.

Status: **experimental**. Not selected by `auto` or any default. Without
`--layout-planner` **or** without an API key, conversion output is
**byte-identical** to main.

## What it does

Given existing block ids, bboxes, and OCR/native text, the planner returns:

| Field | Meaning |
|-------|---------|
| `role` | `title`, `subtitle`, `body`, `list_item`, `caption`, `header`, `footer`, `page_number`, `table_cell`, `other` |
| `reading_order` | Shape / tab order |
| `groups` | Merge/split decisions referencing **existing block ids only** (paragraph / text frame) |
| `font_family_class` | `sans` (黑體), `serif` (明體), `rounded` (圓體), `mono`, `handwriting` + confidence |

The VLM **must not invent text**. Merged frames concatenate OCR/native strings
from the referenced ids. Unknown ids, overlapping groups, malformed JSON, and
low confidence are rejected.

Results are applied through SlideIR:

- `pages[].groups` / `pages[].plan_metadata` (and optional deck-level `plan_metadata`)
- reading order → shape order
- title role → title styling / placeholder
- header / footer / page_number → filtered or kept per `--keep-header-footer`
- paragraph groups → fewer, correctly merged text frames
- font class → configurable PowerPoint font (replaces the OCR `font_family=None` path when the planner succeeds)

## CLI / API

```bash
# Default — geometric heuristics only (byte-identical to main)
pdf2ppt input.pdf output.pptx

# Experimental opt-in Gemini layout planner
export GEMINI_API_KEY=...   # or GOOGLE_API_KEY
pdf2ppt input.pdf output.pptx --layout-planner gemini

# Cost / latency brakes
pdf2ppt input.pdf output.pptx --layout-planner gemini \
  --layout-planner-max-calls 30 \
  --layout-planner-max-cost-usd 1.0 \
  --layout-planner-timeout-s 45 \
  --layout-planner-concurrency 2 \
  --layout-planner-cache-dir .pdf2ppt_cache/layout_planner
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--layout-planner {gemini}` | off | Master switch (not in auto/default) |
| `--layout-planner-model` | `gemini-3.1-flash-lite` | Override via flag or `PDF2PPT_LAYOUT_PLANNER_MODEL` |
| `--layout-planner-max-calls` | `30` | Per-document paid call cap |
| `--layout-planner-max-cost-usd` | `1.0` | Estimated USD budget |
| `--layout-planner-timeout-s` | `45` | Per-call HTTP timeout |
| `--layout-planner-concurrency` | `2` | Concurrent API calls |
| `--layout-planner-cache-dir` | `.pdf2ppt_cache/layout_planner` | Disk cache (image hash + prompt version + model) |
| `--layout-planner-min-confidence` | `0.55` | Reject low-confidence annotations / groups |

HTTP `POST /jobs/{id}/convert` accepts the same fields (`layout_planner`,
`layout_planner_model`, …). The review UI exposes an experimental
“Layout Planner” select. API keys are read from the server environment and are
**never** logged or printed.

## Report fields

When the planner runs, `*.report.json` includes a `layout_planner` object:

- `calls`, `cache_hits`, `fallbacks`
- `estimated_cost_usd`, `total_latency_ms`
- per-page records (`success`, `fallback_kind`, …)

## Evaluation

Annotated Phase-0 fixtures: `tests/fixtures/*.planner.json`.

```bash
# CI-safe recorded responses (no API key)
python scripts/bench_layout_planner.py --mode recorded --compare-heuristic

# Live (cheap): requires GEMINI_API_KEY; uses cache + call cap
python scripts/bench_layout_planner.py --mode live --max-calls 6
```

Acceptance metrics (roadmap Phase 3):

- title detection accuracy
- header/footer filtering rate
- reading-order correctness

The script also surfaces grouping metrics (and can be paired with
`scripts/bench_slide.py` for editability / text F1 / SSIM regression checks).

Live tests are gated:

```bash
PDF2PPT_RUN_LAYOUT_PLANNER=1 pytest -q tests/test_layout_planner.py -k live
```

CI has no API key and no paddlepaddle requirement for these unit tests
(mocked VLM responses only).

## Design invariants

1. **Opt-in only** — never default, never `auto`.
2. **Structured JSON, never pixels** — no hallucination into the slide image.
3. **Fail closed** — any error → geometric heuristics for that page.
4. **No text invention** — planner may only reference existing block ids.
5. **Byte-identical when off / unconfigured** — missing flag or missing API key
   leaves the P0–P5 / refine pipelines unchanged.
