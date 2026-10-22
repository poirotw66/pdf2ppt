# Layout planner recorded responses

Commit **real** Gemini `generateContent` JSON payloads here as
`<fixture>.response.json` (the object returned by the API, or at least the
parsed `{blocks, groups}` plan with **live block ids** such as
`native_1_0_0`).

Do **not** check in ground-truth copies from `*.planner.json`. Those ids
(`title`, `li1`, …) never match live blocks, so the bench rebuilds roles
from GT and silently scores an oracle.

To capture live responses for later commit:

```bash
export GEMINI_API_KEY=...
python scripts/bench_layout_planner.py --mode live --record-responses \
  --recorded-dir tests/fixtures/layout_planner_recorded --max-calls 8
```

Until real files are present, `--mode recorded` reports
`oracle_response: true` / skips acceptance means (no invented planner metrics).
