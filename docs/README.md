# Documentation index

Start here if you are new: [quickstart.md](quickstart.md) · root [README](../README.md) · [繁體中文](../README_tw.md).

## User guides

| Doc | Contents |
| --- | --- |
| [quickstart.md](quickstart.md) | Install (pip / pipx / Docker), first convert, troubleshooting |
| [cli-reference.md](cli-reference.md) | All CLI flags, batch mode, exit codes, OCR timeout lane |
| [review-ui.md](review-ui.md) | OCR review + Preview before convert, API endpoints, 409/413 |
| [slideir.md](slideir.md) | SlideIR schema, `emit-ir` / `from-ir`, safe zip limits |
| [layout-analysis.md](layout-analysis.md) | Heuristic vs PP-Structure layout engines |
| [layout-planner.md](layout-planner.md) | Gemini VLM planner: USD/call caps, circuit breaker, grouping |
| [vector-nativization.md](vector-nativization.md) | Shapes, ruled / borderless / zebra tables, tables over photos |
| [background-engines.md](background-engines.md) | Inpaint engines and `auto` routing |
| [generative-inpaint.md](generative-inpaint.md) | Cloud generative backends (OpenAI / Gemini) |
| [slide-quality.md](slide-quality.md) | Fidelity / editability metrics and `--refine` |

## Planning / historical

These describe design history and may lag the product docs above. Prefer the
user guides when behaviour conflicts.

| Doc | Contents |
| --- | --- |
| [spec.md](spec.md) | Original product spec |
| [plan.md](plan.md) | Implementation plan notes |
| [optimization-roadmap.md](optimization-roadmap.md) | Older optimization roadmap |

## Related root files

| File | Contents |
| --- | --- |
| [CHANGELOG.md](../CHANGELOG.md) | Keep a Changelog history |
| [.env.example](../.env.example) | Compose / API env template (no secrets) |
| [Makefile](../Makefile) | `make help` for lint / test / Docker shortcuts |
