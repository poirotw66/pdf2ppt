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
editing structure (`groups`, `plan_metadata`, roles, reading order) over
rewriting strings.

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

Produces:

```
OUTPUT.slideir.json
OUTPUT.slideir.assets/pages/001/...
```

### Emit IR + PPTX

```bash
pdf2ppt INPUT.pdf OUTPUT.pptx --emit-ir OUTPUT.slideir.json
```

### From IR → PPTX

```bash
pdf2ppt from-ir OUTPUT.slideir.json REBUILT.pptx
```

Keep the `.assets` directory next to the JSON (or zip it with the same layout
for `POST /from-ir`).

## Editing rules

- **Safe:** adjust `reading_order`, `block_role`, `layout_label`, `groups`,
  `plan_metadata`; tweak shape/table styling the renderer already supports.
- **Unsafe:** fabricate new text content; invent shape kinds outside the schema
  enum; point `asset.path` outside `assets_dir`.
- After edits, re-run `from-ir` and visually QA the PPTX.

## HTTP (optional)

- `POST /jobs/{id}/emit-ir` — IR only  
- `POST /jobs/{id}/convert` with `{"emit_ir": true, "ir_only": false}`  
- `POST /from-ir` — multipart `ir_file` (+ optional `assets_zip`) → PPTX

## Round-trip check

```bash
pdf2ppt emit-ir tests/fixtures/solid_background.pdf /tmp/deck.slideir.json
pdf2ppt from-ir /tmp/deck.slideir.json /tmp/deck.pptx
```
