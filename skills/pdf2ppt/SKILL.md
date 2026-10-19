---
name: pdf2ppt
description: >-
  Use pdf2ppt as a deterministic evidence extractor and native PPTX renderer via
  SlideIR. Prefer emit-ir → edit IR → from-ir over re-running the full PDF
  pipeline when reconstructing or refining editable decks.
---

# pdf2ppt SlideIR skill

pdf2ppt converts PDFs to editable PPTX. For agent workflows, treat it as two
tools around **SlideIR** (JSON + sibling asset files):

1. **emit-ir** — run analysis (layout, OCR, vector nativization) and write IR.
2. **from-ir** — load IR and render native PPTX (no PDF).

Do **not** invent text; keep `text_blocks[].text` as ground truth. Prefer
editing roles, reading order, and renderer-supported styles over rewriting
strings.

## Prerequisites

```bash
python -m pip install -e .
# optional: same OCR/layout models as normal convert
```

Schema: `src/pdf2ppt/schemas/slideir.schema.json`  
Docs: `docs/slideir.md`

## Commands

### Emit IR (no PPTX)

```bash
pdf2ppt emit-ir INPUT.pdf OUTPUT.slideir.json \
  --vector-mode auto \
  --layout-engine heuristic
```

Produces (assets dir is `{ir_filename}.assets`):

```
OUTPUT.slideir.json
OUTPUT.slideir.json.assets/pages/001/...
```

### Emit IR + PPTX

```bash
pdf2ppt INPUT.pdf OUTPUT.pptx --emit-ir OUTPUT.slideir.json
```

### From IR → PPTX

```bash
pdf2ppt from-ir OUTPUT.slideir.json REBUILT.pptx
```

Keep the `{ir_filename}.assets` directory next to the JSON (or zip it with the
same layout for `POST /from-ir`).

## Editing rules

- **Safe (affects PPTX today):** adjust `reading_order`, `block_role`,
  `layout_label`; tweak shape/table/text styling fields the renderer already
  supports (colors, sizes, bboxes it already maps).
- **Forward-compatible only (not rendered today):** `groups` and
  `plan_metadata` are reserved for P3 planners. You may leave or add them for
  later tools, but `from-ir` / `ppt_render` currently ignore them — editing
  them will not change the PPTX.
- **Unsafe:** fabricate new text content; invent shape kinds outside the schema
  enum; point `asset.path` outside `assets_dir`.
- After edits, re-run `from-ir` and visually QA the PPTX.

## HTTP (optional)

- `POST /jobs/{id}/emit-ir` — IR only  
- `GET /jobs/{id}/slideir.json` / `GET /jobs/{id}/slideir-assets.zip` — download emit-ir artifacts  
- `POST /jobs/{id}/convert` with `{"emit_ir": true, "ir_only": false}`  
- `POST /from-ir` — multipart `ir_file` (+ optional `assets_zip`) → PPTX

## Round-trip check

```bash
pdf2ppt emit-ir tests/fixtures/solid_background.pdf /tmp/deck.slideir.json
pdf2ppt from-ir /tmp/deck.slideir.json /tmp/deck.pptx
```
