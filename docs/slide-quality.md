# Slide-quality evaluation (Phase 0 / P0)

This document describes how to measure whether a pdf2ppt change improved
**fidelity** (does the rendered slide look like the source page?) and
**editability** (how much is native / editable vs. a raster background?).

The two scores are always reported separately. A full-page background image can
look nearly identical to the source while remaining hard to edit.

## Why this exists

`docs/optimization-roadmap.md` Phase 0 notes that quality could not be
quantified. Future work (vector nativization, generative inpainting, VLM
planning) needs a stable, diffable JSON baseline. This evaluator is that
baseline; it does **not** change default conversion behaviour.

## Components

| Piece | Role |
|-------|------|
| `src/pdf2ppt/render_compare.py` | Metrics + LibreOffice render path |
| `scripts/bench_slide.py` | Convert fixtures → score → JSON |
| `tests/test_render_compare.py` | Unit tests for metrics (LibreOffice tests skip if missing) |

Related but distinct: `scripts/bench_inpaint.py` scores **inpainting engines**
on mask cleanliness. The slide evaluator scores the **end-to-end PPTX**.

## Metrics

### Fidelity (rendered slide vs. source PDF page)

1. **SSIM** — structural similarity on grayscale (numpy/OpenCV; no scikit-image).
2. **Pixel MAE / MSE** — mean absolute / squared RGB error on a 0–255 scale.
3. **Edge F1** — Canny edge-map precision/recall F1.
4. **Text F1** — bag-of-tokens F1: native PDF text (ground truth) vs. OCR on the
   rendered slide. `null` when OCR is unavailable (`--ocr off` / auto-degrade).

### Editability (from PPTX structure; no render required)

- Counts: text boxes, pictures, auto-shapes, tables, lines, groups, charts.
- **`raster_background_area_ratio`** — largest picture covering ≥ 85% of the
  slide area (typical full-page overlay / full-page background).
- **`score`** — composite in `[0, 1]` that rewards native objects and penalizes
  full-page rasters. Use the raw counts for fine-grained diffs; use `score` for
  a single comparable number.

## System dependencies (optional)

Conversion itself does not need these. Only evaluation does:

```bash
# Debian / Ubuntu
sudo apt-get install -y libreoffice-impress fonts-noto-cjk
```

- **LibreOffice** (`soffice`) converts PPTX → PDF headlessly; pages are then
  rasterized with PyMuPDF (same stack as the pipeline).
- **Noto Sans CJK TC** — pdf2ppt already writes this family into OOXML for CJK
  text (`text_style.default_font_family`). Pin the same font for rendering so
  metrics are reproducible across machines.

### LibreOffice ≠ PowerPoint

Font metrics, CJK layout, and anti-aliasing differ between LibreOffice and
Microsoft PowerPoint. Treat these scores as a **reproducible CI / laptop
baseline**, not as a WYSIWYG guarantee of what PowerPoint users will see.
Compare runs on the same renderer + font set.

## How to run

```bash
# Metric unit tests (no LibreOffice required)
python -m pytest tests/test_render_compare.py -q

# Full fixture benchmark (needs LibreOffice)
python scripts/bench_slide.py --ocr off
python scripts/bench_slide.py --json-out output/bench_slide.json

# Digital-only fixtures (no PaddleOCR needed during conversion)
python scripts/bench_slide.py --ocr off \
  --fixture solid_background --fixture dark_background_light_text \
  --fixture geometric_chart --fixture table

# Alternate conversion knobs (local + opt-in generative inpainting)
python scripts/bench_slide.py --inpaint-engine white-box
python scripts/bench_slide.py --inpaint-engine lama-onnx --extra-option inpaint_model_root=model/lama
# Requires OPENAI_API_KEY / GEMINI_API_KEY; see docs/generative-inpaint.md
python scripts/bench_slide.py --inpaint-engine generative-openai \
  --extra-option generative_max_calls=5 --extra-option generative_max_cost_usd=1.0
```

Hybrid fixtures (`gradient_*`, `photo_texture`, `large_mask`) still run
`convert_pdf`, which needs a working PaddleOCR + PaddlePaddle stack. The bench
defaults `--convert-ocr-model-root` to `None` (official PaddleX cache) so an
empty local `model/` directory does not break those cases.

JSON layout (abbreviated):

```json
{
  "cases": [
    {
      "name": "solid_background",
      "pages": [
        {
          "page_number": 1,
          "fidelity": { "ssim": 0.91, "pixel_mae": 4.2, "edge_f1": 0.55, "text_f1": null },
          "editability": { "text_box_count": 3, "raster_background_area_ratio": 1.0, "score": 0.31 }
        }
      ]
    }
  ],
  "summary": { "fidelity": { "ssim_mean": 0.88 }, "editability": { "score_mean": 0.29 } }
}
```

Diff two runs by comparing `summary` and per-fixture `pages[*].fidelity` /
`pages[*].editability` in the JSON files.

## CI

The `slide-quality` job in `.github/workflows/ci.yml` is **report-only**
(`continue-on-error: true`). It installs LibreOffice + Noto CJK, runs the
metric tests, and attempts a short fixture bench with `--ocr off`. Failures
there never block merge; the blocking `backend` job still runs the pure metric
unit tests (LibreOffice cases skip automatically).

## Interface stability for future backends

`evaluate_pptx_against_pdf(pptx, pdf, *, dpi, ocr_engine=None)` only needs the
output PPTX and source PDF. `scripts/bench_slide.py --inpaint-engine …` forwards
conversion knobs into `ConversionOptions`. Measure the generative backends with
`--inpaint-engine generative-openai` or `generative-gemini` (plus budget extras);
see [`generative-inpaint.md`](./generative-inpaint.md).
