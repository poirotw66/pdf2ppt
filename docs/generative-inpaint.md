# Generative (cloud) image-edit inpainting

Opt-in backend that uses modern image-generation / image-edit models to regenerate
**masked text regions** when classic OpenCV / LaMa inpainting leaves artifacts on
photos or complex illustrations.

This is Phase 4 of [`optimization-roadmap.md`](./optimization-roadmap.md). It is
**never** selected by `--inpaint-engine auto` and does nothing when no API key is
configured.

## Providers and default model IDs

| Engine flag | Provider | Default model | Official API |
|-------------|----------|---------------|--------------|
| `generative-openai` | OpenAI Images Edit | `gpt-image-2` | `POST /v1/images/edits` (image + alpha mask) |
| `generative-gemini` | Google Gemini / Nano Banana | `gemini-nano-banana-2.1` | `POST /v1beta/interactions` (conversational image edit) |

Override models with `--generative-model`, or:

- `PDF2PPT_OPENAI_IMAGE_MODEL`
- `PDF2PPT_GEMINI_IMAGE_MODEL`

Credentials (read from the environment; never hard-code):

- OpenAI: `OPENAI_API_KEY`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`

## Mask-only compositing

Regardless of what the model returns:

1. pdf2ppt sends a **cropped patch** (with context padding) plus a mask of the text region.
2. The model output is **hard-composited** back onto the original crop so every
   unmasked pixel stays byte-identical to the source.
3. Patches are written back onto the full page the same way.

OpenAI receives an RGBA mask where **alpha = 0** means “edit here” (official
Images Edit convention). Gemini has no mask parameter, so the masked pixels are
blanked before the request; compositing still enforces the mask lock.

## Safety, budget, and cache

After each generative patch:

- **OCR residual check** (default on): re-run the existing OCR detector over the
  patched region. If text still overlaps the mask (or new text was hallucinated
  into it), that patch falls back to local `opencv-fast`.
- **API / network errors** also fall back to local inpainting for that patch.
- **Per-run budget**: `--generative-max-calls` (default 50) and
  `--generative-max-cost-usd` (default 5.0). Further patches fall back locally.
- **Disk cache**: keyed by `sha256(provider + model + prompt + image + mask)` under
  `--generative-cache-dir` (default `.pdf2ppt_cache/generative_inpaint`).

Call counts, cache hits, fallbacks, and estimated USD cost are written into the
conversion report under `generative_inpaint`.

## Cost and non-determinism

- Generative edits are **non-deterministic**: the same page can look different
  across runs. Prefer local engines when you need reproducibility.
- Estimated costs are **guardrails**, not invoices. Rough public list prices used
  for budgeting (verify on the vendor pricing pages before production use):
  - OpenAI GPT Image edits: on the order of several cents per call (token-based).
  - Gemini Nano Banana 2.1 (`gemini-nano-banana-2.1`): about **$0.0336** per 1K
    output image (standard tier, image-output tokens).
- A dense deck can produce many patch groups. Keep the call/cost caps low for
  interactive review; raise them only for batch jobs you are willing to pay for.

## CLI example

```bash
export OPENAI_API_KEY=sk-...
pdf2ppt input.pdf output.pptx \
  --inpaint-engine generative-openai \
  --generative-max-calls 20 \
  --generative-max-cost-usd 2.0 \
  --report output.report.json
```

```bash
export GEMINI_API_KEY=...
pdf2ppt input.pdf output.pptx \
  --inpaint-engine generative-gemini \
  --generative-model gemini-nano-banana-2.1
```

Without an API key, an explicit `generative-*` engine **soft-falls back** to
`opencv-fast` / `white-box` so conversion still completes. Default engines and
`auto` are completely unchanged when keys are absent.

## Review UI

The review sidebar lists `generative-openai` and `generative-gemini` in the
inpaint-engine dropdown. Treat **Convert PPTX** as the regenerate action for the
current reviewed boxes: pick a generative engine, ensure the API server has the
matching env key, then convert. There is no separate per-box regenerate button
yet; page-level regenerate is the intentional Phase 4 entry point.

## Benchmarking with the slide evaluator

`scripts/bench_slide.py` already forwards `--inpaint-engine` into
`ConversionOptions`, so generative backends can be measured with the same
fidelity / editability JSON as local engines:

```bash
python scripts/bench_slide.py --ocr off \
  --inpaint-engine generative-openai \
  --extra-option generative_max_calls=5 \
  --extra-option generative_max_cost_usd=1.0
```

See [`slide-quality.md`](./slide-quality.md).

## Optional image restore (benchmark hook)

`GenerativeInpaintingEngine.restore_image(image)` is a separate callable for
mild quality restoration of embedded bitmaps. It is **not** wired into the
default conversion path; keep it available for later render-and-compare
benchmarks.

## License note

Do **not** copy code from AGPL projects such as banana-slides. This backend is
an independent MIT-licensed implementation that talks to the vendors’ public
APIs.
