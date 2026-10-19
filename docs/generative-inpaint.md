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
4. Generative composites apply a mild **boundary color match** and edge feather so
   patched rectangles are less visible on gradients.

OpenAI receives an RGBA mask where **alpha = 0** means “edit here” (official
Images Edit convention). Gemini has no mask parameter, so the masked pixels are
blanked before the request; compositing still enforces the mask lock.

### Gemini REST response shape

The Interactions API HTTP JSON does **not** include the SDK convenience field
`output_image`. Generated images arrive under `steps[].content[]` items with
`type: "image"` and base64 `data` (see [Getting started](https://ai.google.dev/gemini-api/docs/get-started)
and [Interactions API](https://ai.google.dev/api/interactions-api-v1)). The
provider takes the **last** image block in `steps`.

### Gemini aspect ratio

Nano Banana chooses a discrete output aspect ratio. pdf2ppt pads each crop to the
nearest supported ratio (`1:1`, `16:9`, `4:1`, …), requests that ratio via
`response_format.aspect_ratio`, then crops the content window back **without**
stretching long thin patches.

### Text mask dilation

Each text box is dilated by a fraction of its **line height** (default **18%**,
clamped by `--inpaint-mask-dilate-min-px` / `--inpaint-mask-dilate-max-px`).
`--inpaint-mask-dilate-px` (default **2**) is an **absolute floor** on top of
that, so anti-aliased glyph tops/bottoms stay covered when OCR boxes are
vertically tight. Adjacent dilated islands are morphologically closed/merged
before cropping so generative patch counts do not explode (e.g. solid pages
staying on a single full-page call). This applies to **all** inpaint engines.

### Poisson blending and low-frequency fallback

Generative patches are composited with `cv2.seamlessClone` using
**`NORMAL_CLONE`** (not `MIXED_CLONE`). Mixed mode keeps the stronger of
source/destination gradients and was pulling original glyph edges back into
clean generative fills, which then failed the OCR residual check. After
compositing, pdf2ppt applies a **texture-aware** low-frequency gate:

- If the **outside ring** around the mask is flat (luma std ≤
  `--generative-flat-ring-luma-std`, default **4.0**), the generative API is
  **skipped entirely** and opencv-fast is used (`flat_ring_skips` in the report).
  Smooth slides are where classic inpainting already wins — no spend.
- On mid-range rings the base `--generative-low-freq-mae-threshold` (default
  **4.0**) is scaled up by surrounding variance.
- On highly textured rings (luma std ≥
  `--generative-textured-ring-luma-std`, default **16.0**) the low-freq gate is
  skipped so photo backgrounds are not rejected for high-frequency mismatch.

## Safety, budget, and cache

After each generative patch:

- **OCR residual check** (default on): re-run the existing OCR detector over the
  patched region. If text still overlaps the mask (or new text was hallucinated
  into it), that patch falls back to local `opencv-fast`.
- **API / network errors** also fall back to local inpainting for that patch.
- **Copyright / recitation / safety blocks** (e.g. Gemini
  `copyright/recitation`) are classified as **expected safety-block fallbacks**,
  not API outages. Because these blocks are nondeterministic, pdf2ppt **retries
  once** by default (`--generative-safety-block-retries`, default 1) before
  falling back. Official responses that say the request is not charged are
  recorded at **0 USD**; retries are counted in `safety_block_retries` and
  **do not consume** `--generative-max-calls` (only billed calls do).
- **Flat-ring skip** and **low-frequency quality gate** (above) can also fall
  back to opencv-fast.
- Local fallback results are **re-composited through the mask** so unmasked
  pixels stay unchanged even when `opencv-fast` would otherwise touch them.
- **Per-run budget**: `--generative-max-calls` (default 50) counts **billed**
  calls only; `--generative-max-attempts` (default 3× max_calls) caps total
  provider attempts including free safety blocks; `--generative-max-cost-usd`
  (default 5.0) caps estimated USD.
- **Disk cache**: keyed by `sha256(provider + model + prompt + image + mask)` under
  `--generative-cache-dir` (default `.pdf2ppt_cache/generative_inpaint`).

Call counts (`billed_calls` / `attempts`), cache hits, fallbacks
(`safety_block_fallbacks`, `api_error_fallbacks`, `low_freq_fallbacks`,
`flat_ring_skips`), `safety_block_retries`, estimated USD cost, and
`cost_uncertain` are written into the conversion report under
`generative_inpaint`.

## Cost and non-determinism

- Generative edits are **non-deterministic**: the same page can look different
  across runs. Prefer local engines when you need reproducibility.
- Costs are computed from provider **usage** metadata when available, using the
  rates in `src/pdf2ppt/generative_inpaint/pricing.py` (single config place):
  - Gemini Nano Banana 2.1: **$1.50 / M** input, **$7.50 / M** text+thinking
    output, **$30 / M** image output (≈ **$0.038** per typical 1K edit when
    usage is missing). When `output_tokens_by_modality` lists only image tokens
    but `total_output_tokens` is higher, the leftover (~256 text tokens) is
    priced at the text+thinking rate.
  - OpenAI GPT Image edits: text/image input + image output token rates in the
    same module; flat **$0.08** estimate when usage is missing.
- Failed calls that may still have been billed (e.g. HTTP 200 with a missing
  image payload) record the estimate and set `cost_uncertain=true`. Safety
  blocks that are not charged stay at 0.
- A dense deck can produce many patch groups. Keep the call/cost caps low for
  interactive review; raise them only for batch jobs you are willing to pay for.

## Default prompt

The shared default prompt is intentionally **neutral** (“continue the slide
background / fill blacked-out areas”). Wording that asks to remove logos or
watermarks has triggered Gemini safety filters in real runs; both OpenAI and
Gemini use the same default.

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
