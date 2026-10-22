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
**byte-identical** to main (PPTX zip member contents). A flag-without-key run
records `layout_planner.skip_reason` in the report.

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

### Default application (low-risk)

By default the planner **keeps heuristic paragraph/list grouping exactly as
planner-off**, then layers only:

- roles / title placeholders
- reading order
- header / footer / page_number filtering (OCR sources, when `--keep-header-footer` is off)

It does **not** replace frames with one-box-per-line. That regression split
numbered lists into separate auto-numbered boxes (`1. 1. 1. …`).

**Grouping merges** and **font-class remaps** are **opt-in**
(`--layout-planner-apply-grouping`, `--layout-planner-apply-fonts`). Live
verification showed unrestricted merges collapsing tables/charts and font
remaps reflowing digital Helvetica/Times pages.

When fonts are enabled:

- Keep the PDF/OCR `font_family` when it is a real name (never replace Helvetica → DejaVu).
- Apply the VLM class only when the source font is unknown (typical scanned/OCR pages).
- Map Latin to Arial / Times New Roman / Courier New; CJK to **Noto Sans CJK TC**
  (fallback list also documents PingFang TC / Microsoft JhengHei / PMingLiU —
  PowerPoint’s east-asian slot substitutes when a face is missing).
- Use a CJK font only when the text contains CJK characters (Latin+digit lines stay Latin).
- Never change font size or bbox (no reflow from planner fonts).

Grouping (when enabled) only merges **multi-member** groups that pass safety
rules (vertically adjacent, left/indent-aligned, similar font size, one column);
rejects `table_cell` / table-or-chart obstacles; keeps each line’s original
geometry via `line_bboxes`. Single-member groups use the normal per-block path
and **never** pin font size (so shrink-to-fit still works).

## CLI / API

```bash
# Default — geometric heuristics only (byte-identical to main)
pdf2ppt input.pdf output.pptx

# Experimental opt-in Gemini layout planner (low-risk apply)
export GEMINI_API_KEY=...   # or GOOGLE_API_KEY
pdf2ppt input.pdf output.pptx --layout-planner gemini

# Opt into grouping / fonts (higher risk for SSIM on tables & digital pages)
pdf2ppt input.pdf output.pptx --layout-planner gemini \
  --layout-planner-apply-grouping \
  --layout-planner-apply-fonts

# Cost / latency brakes
pdf2ppt input.pdf output.pptx --layout-planner gemini \
  --layout-planner-max-calls 30 \
  --layout-planner-max-cost-usd 1.0 \
  --layout-planner-timeout-s 45 \
  --layout-planner-concurrency 2 \
  --layout-planner-max-consecutive-failures 2 \
  --layout-planner-cache-dir .pdf2ppt_cache/layout_planner
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--layout-planner {gemini}` | off | Master switch (not in auto/default) |
| `--layout-planner-model` | `gemini-3.1-flash-lite` | Override via flag or `PDF2PPT_LAYOUT_PLANNER_MODEL` |
| `--layout-planner-max-calls` | `30` | Per-document paid call cap (**every HTTP attempt** counts, including failures/retries) |
| `--layout-planner-max-cost-usd` | `1.0` | Estimated USD budget |
| `--layout-planner-timeout-s` | `45` | Per-call HTTP timeout |
| `--layout-planner-concurrency` | `2` | Concurrent API calls |
| `--layout-planner-cache-dir` | `.pdf2ppt_cache/layout_planner` | Disk cache (image hash + prompt version + model) |
| `--layout-planner-min-confidence` | `0.55` | Reject low-confidence annotations / groups |
| `--layout-planner-apply-grouping` | off | Opt into paragraph merges |
| `--layout-planner-apply-fonts` | off | Opt into VLM font class when PDF font unknown |
| `--layout-planner-max-consecutive-failures` | `2` | Open circuit after N consecutive failures; auth errors open immediately |

HTTP `POST /jobs/{id}/convert` accepts `layout_planner`, clamped
`layout_planner_max_calls` / `layout_planner_timeout_s`, and the apply flags.
**Model id, cache directory, and USD cost caps are server-side only** (env /
process defaults) — clients cannot choose them. API keys are read from the
server environment and are **never** logged or printed.

## Budget, circuit breaker, concurrency

- Slots are **reserved atomically** before the HTTP call (`try_reserve`) so
  concurrent workers cannot overshoot `max_calls`.
- Failed requests (bad JSON, timeouts, API/auth errors) still count toward
  `calls` / `http_attempts` / estimated cost.
- Cost uses Gemini `usageMetadata` token counts with configurable `$/1M`
  input/output prices (`PDF2PPT_LAYOUT_PLANNER_INPUT_USD_PER_MTOK` /
  `…_OUTPUT_…`, defaults $0.25 / $1.50). The dollar cap reserves a **worst-case
  next-call** cost from generous prompt tokens + `maxOutputTokens` (default
  8k in + 8192 out ≈ $0.0143) and reconciles to actual usage after each
  response, so a single expensive call cannot silently overshoot the cap.
  Failed calls refund unused reserve (flat estimate when usage is unknown).
  A warning is logged when `max_cost_usd` is below one reserve.
- Prompt evidence is capped (block count + per-block characters) so a dense
  page cannot inflate the request past the reserved worst case.
- After `max_consecutive_failures` consecutive failures, or any auth error,
  the **circuit opens** and remaining pages fall back without waiting on the
  full timeout.
- Text-regression fallback allows intended OCR header/footer/page_number
  removals by matching heuristic frames back to plan source ids (`ocr_*` /
  `native_*`); only non-HF body loss triggers a per-page fallback.

## Report fields

When the planner runs (or is requested), `*.report.json` includes a
`layout_planner` object:

- `calls`, `http_attempts`, `cache_hits`, `fallbacks`
- `estimated_cost_usd`, `total_latency_ms`
- `circuit_open`, `circuit_reason`
- `apply_grouping`, `apply_fonts`
- `skip_reason` when the flag is set but no API key / transport is configured
- per-page records (`success`, `fallback_kind`, `http_attempted`, …)

## Evaluation

Annotated fixtures: `tests/fixtures/*.planner.json` (English lists/columns,
table, chart, scanned list, mixed CJK+English).

```bash
# Recorded mode needs committed *live* captures (real block ids). Missing /
# GT-copied files are skipped as oracle and excluded from averages.
python scripts/bench_layout_planner.py --mode recorded --compare-heuristic

# Live (cheap): requires GEMINI_API_KEY; optionally write captures to commit later
python scripts/bench_layout_planner.py --mode live --max-calls 8 --compare-heuristic \
  --record-responses --recorded-dir tests/fixtures/layout_planner_recorded
```

Acceptance metrics (from the **output PPTX**, not GT-vs-GT):

- title detection accuracy
- header/footer filtering rate (HF strings absent from editable text)
- reading-order correctness

Committed real captures under `tests/fixtures/layout_planner_recorded/`
(replay of live `gemini-3.1-flash-lite` responses) currently score
**title 0.964**, **HF 0.75**, **reading order 0.944**. Quote recorded
metrics only from these live-origin files.

`--compare-heuristic` adds per-page **SSIM** (when LibreOffice is available),
**text F1**, and **editability** deltas vs planner-off. Default apply flags
should show no regressions beyond noise. Mixed title+body heuristic frames
must not use the title-only placeholder (that would drop body lines); pages
that still lose non-HF text vs planner-off fall back for that page.

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
6. **Low-risk apply by default** — grouping and fonts behind explicit flags.
7. **No silent text loss** — title placeholder / title role only when every
   source line in the frame is a title; renderer never drops placeholder text;
   IR text-regression vs planner-off triggers a per-page fallback.
