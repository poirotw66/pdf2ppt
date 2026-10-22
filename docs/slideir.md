# SlideIR (P5 agent ecosystem)

SlideIR is pdf2ppt’s versioned JSON intermediate representation. Agents use it
as a **deterministic evidence layer**: extract once (`emit-ir`), optionally
edit structure fields, then render (`from-ir`) without re-running the PDF
pipeline.

Schema file (bundled): `src/pdf2ppt/schemas/slideir.schema.json`  
Python I/O: `pdf2ppt.ir_io`

## What is in scope

IR ⊆ what P1/P2 already render:

| Field | Source |
|-------|--------|
| Text blocks (id, bbox, text, style, role, reading order, paragraphs, line_bboxes) | native + OCR + layout |
| Shapes (`is_page_background`, `render_under_raster`, …) | vector nativization + paint-order promotion |
| Tables | `find_tables()` |
| Pictures / crops | image elements + figure crops |
| Layout regions | heuristic / PP-Structure |
| Background mode + raster | overlay / full-page / elements |
| Groups / `plan_metadata` | optional; Phase-3 layout planner — see below |

Text is treated as ground truth (`locked: true` on emit). Do not invent geometry
the renderer cannot express yet.

With `--layout-planner gemini`, a successful plan writes `pages[].plan_metadata`
and applies block roles and reading order to `text_blocks` before the IR is
saved (reading order → shape order, title roles → title placeholder styling,
header/footer/page_number filtered per `--keep-header-footer`). Paragraph
merges and `pages[].groups` are written only with
`--layout-planner-apply-grouping`. Deck-level `plan_metadata` is written
whenever `--layout-planner` is enabled (including when pages fall back). Font
class → configured font family requires `--layout-planner-apply-fonts`. See
[`layout-planner.md`](./layout-planner.md).

## Asset convention

**Relative paths next to the IR file** (not base64):

```
deck.slideir.json
deck.slideir.json.assets/
  pages/
    001/
      background.png  # optional; .png/.jpg depending on image data
      image_000.png
```

- `assets_dir` in the JSON names the sibling directory (default:
  `{ir_filename}.assets`, e.g. `deck.slideir.json.assets`).
- Asset `path` values are relative to that directory (e.g.
  `pages/001/background.png`).
- Paths must not contain `..` or be absolute.

## CLI

```bash
# Emit IR only (no PPTX)
pdf2ppt emit-ir input.pdf deck.slideir.json
# equivalent:
pdf2ppt input.pdf --emit-ir deck.slideir.json --ir-only

# Convert + also write IR
pdf2ppt input.pdf output.pptx --emit-ir deck.slideir.json

# Render IR → PPTX (no PDF)
pdf2ppt from-ir deck.slideir.json output.pptx
# equivalent:
pdf2ppt --from-ir deck.slideir.json -o output.pptx
```

Analysis flags (`--vector-mode`, `--layout-engine`, OCR/inpaint options, …)
apply to `emit-ir` the same way as `convert`.

## HTTP API (opt-in)

| Endpoint | Behavior |
|----------|-----------|
| `POST /jobs/{id}/convert` with `emit_ir` / `ir_only` | Same convert options; optionally write SlideIR |
| `POST /jobs/{id}/emit-ir` | Analysis → SlideIR only (approved boxes optional) |
| `GET /jobs/{id}/slideir.json` | Download emitted SlideIR JSON |
| `GET /jobs/{id}/slideir-assets.zip` | Download the sibling assets directory as a zip |
| `POST /from-ir` | Multipart: `ir_file` + optional `assets_zip` → PPTX download |

`POST /jobs/{id}/emit-ir` needs a job created first via `POST /jobs` with a
JSON body (`{}` is fine). IR downloaded from the API names its assets folder
`output.slideir.json.assets`, so CLI `from-ir` users must unzip into a folder
of that name (`POST /from-ir` handles it automatically).

Defaults leave existing convert behavior unchanged.

## Agent workflow

1. `emit-ir` on a PDF → valid SlideIR + assets.
2. Edit JSON (roles, reading order, styles the renderer already supports)
   without rewriting text. Default `--layout-planner gemini` writes
   `plan_metadata` and applies roles / reading order; paragraph merges and
   `pages[].groups` require `--layout-planner-apply-grouping`. `from-ir`
   renders the structures already present in the IR.
3. `from-ir` → schema-valid PPTX via `ppt_render`.

See `skills/pdf2ppt/SKILL.md` for a Codex/Claude-oriented skill pack.
