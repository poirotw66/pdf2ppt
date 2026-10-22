# Layout planner recorded responses

These `*.response.json` files are **real** Gemini `generateContent` captures
from `--mode live --record-responses` (model `gemini-3.1-flash-lite`).
They contain only block ids / roles / reading order / confidence /
`font_family_class` — no OCR text and no API key. Each file sets
`recorded_from: live` plus `model` metadata.

Replaying them with `--mode recorded` (no API key) reproduces the live
acceptance means from that capture:

| Metric | Value |
|--------|------:|
| title detection | 0.964 |
| header/footer filter | 0.75 |
| reading order | 0.944 |

Do **not** check in ground-truth copies from `*.planner.json`. Those ids
(`title`, `li1`, …) never match live blocks, so the bench would rebuild roles
from GT and silently score an oracle. Files missing `recorded_from: live` /
model metadata, or whose overlapping ids mirror GT roles+confidences (including
a GT copy plus one decoy id), are skipped as oracle and excluded from averages.

To refresh captures:

```bash
export GEMINI_API_KEY=...
python scripts/bench_layout_planner.py --mode live --record-responses \
  --recorded-dir tests/fixtures/layout_planner_recorded --max-calls 8
```
