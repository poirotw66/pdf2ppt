"""Preview-before-convert: SlideIR overlay payload and light edit application.

Edits are stored as a job-scoped overlay JSON (not baked into the original
emit-ir output). Applying them produces an edited SlideIR that ``from-ir``
renders, so the preview matches the PPTX.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Literal

from .ir_io import SlideIrError, validate_slide_ir_payload

PREVIEW_EDITS_FILENAME = "preview_edits.json"
EDITED_SLIDEIR_FILENAME = "output.edited.slideir.json"

ALLOWED_BLOCK_ROLES = frozenset({"title", "body", "header", "footer", "page_number"})
MAX_PREVIEW_EDIT_PAGES = 10_000
MAX_EXCLUDED_IDS_PER_PAGE = 20_000
MAX_ROLE_OVERRIDES_PER_PAGE = 20_000
THUMBNAIL_WIDTH_PX = 120

ElementKind = Literal["text", "table", "image", "shape"]
ELEMENT_KIND_TO_ARRAY = {
    "text": "text_blocks",
    "table": "tables",
    "image": "images",
    "shape": "shapes",
}


def empty_preview_edits() -> dict[str, Any]:
    return {"pages": []}


def load_preview_edits(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_preview_edits()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SlideIrError(f"Invalid preview edits JSON: {error}") from error
    if not isinstance(payload, dict):
        raise SlideIrError("Preview edits root must be a JSON object.")
    return normalize_preview_edits(payload)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` via a same-directory temp file + ``os.replace`` (atomic on POSIX)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def save_preview_edits(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_preview_edits(payload)
    atomic_write_text(path, json.dumps(normalized, ensure_ascii=False, indent=2) + "\n")
    return normalized


def normalize_preview_edits(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a preview-edits overlay."""
    pages_raw = payload.get("pages")
    if pages_raw is None:
        pages_raw = []
    if not isinstance(pages_raw, list):
        raise SlideIrError("preview edits 'pages' must be an array.")
    if len(pages_raw) > MAX_PREVIEW_EDIT_PAGES:
        raise SlideIrError(
            f"preview edits 'pages' exceeds the limit of {MAX_PREVIEW_EDIT_PAGES} entries."
        )

    pages: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    for index, item in enumerate(pages_raw):
        if not isinstance(item, dict):
            raise SlideIrError(f"preview edits pages[{index}] must be an object.")
        try:
            page_number = int(item["page"])
        except (KeyError, TypeError, ValueError) as error:
            raise SlideIrError(f"preview edits pages[{index}] requires integer 'page'.") from error
        if page_number < 1:
            raise SlideIrError(f"preview edits pages[{index}].page must be >= 1.")
        if page_number in seen_pages:
            raise SlideIrError(f"Duplicate preview edits page {page_number}.")
        seen_pages.add(page_number)

        excluded_raw = item.get("excluded_ids") or []
        if not isinstance(excluded_raw, list):
            raise SlideIrError(f"pages[{index}].excluded_ids must be an array.")
        if len(excluded_raw) > MAX_EXCLUDED_IDS_PER_PAGE:
            raise SlideIrError(
                f"pages[{index}].excluded_ids exceeds the limit of {MAX_EXCLUDED_IDS_PER_PAGE} entries."
            )
        excluded_ids: list[str] = []
        seen_excluded: set[str] = set()
        for eid in excluded_raw:
            if not isinstance(eid, str) or not eid.strip():
                raise SlideIrError(f"pages[{index}].excluded_ids entries must be non-empty strings.")
            if eid in seen_excluded:
                continue
            seen_excluded.add(eid)
            excluded_ids.append(eid)

        roles_raw = item.get("role_overrides") or {}
        if not isinstance(roles_raw, dict):
            raise SlideIrError(f"pages[{index}].role_overrides must be an object.")
        if len(roles_raw) > MAX_ROLE_OVERRIDES_PER_PAGE:
            raise SlideIrError(
                f"pages[{index}].role_overrides exceeds the limit of {MAX_ROLE_OVERRIDES_PER_PAGE} entries."
            )
        role_overrides: dict[str, str] = {}
        for block_id, role in roles_raw.items():
            if not isinstance(block_id, str) or not block_id.strip():
                raise SlideIrError(f"pages[{index}].role_overrides keys must be non-empty strings.")
            if not isinstance(role, str) or role not in ALLOWED_BLOCK_ROLES:
                raise SlideIrError(
                    f"pages[{index}].role_overrides[{block_id!r}] must be one of "
                    f"{sorted(ALLOWED_BLOCK_ROLES)}."
                )
            role_overrides[block_id] = role

        raster_fallback = bool(item.get("raster_fallback", False))
        pages.append(
            {
                "page": page_number,
                "excluded_ids": excluded_ids,
                "role_overrides": role_overrides,
                "raster_fallback": raster_fallback,
            }
        )

    pages.sort(key=lambda entry: int(entry["page"]))
    return {"pages": pages}


def build_preview_payload(
    slideir: dict[str, Any],
    *,
    job_id: str,
    edits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a UI-friendly preview payload from SlideIR + optional edits overlay."""
    issues = validate_slide_ir_payload(slideir)
    if issues:
        raise SlideIrError("Invalid SlideIR for preview: " + "; ".join(issues[:5]))

    edits_norm = normalize_preview_edits(edits or empty_preview_edits())
    edits_by_page = {int(page["page"]): page for page in edits_norm["pages"]}

    pages_out: list[dict[str, Any]] = []
    for page in slideir.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page_number = int(page["page_number"])
        page_edits = edits_by_page.get(page_number, {})
        excluded = set(page_edits.get("excluded_ids") or [])
        raster_fallback = bool(page_edits.get("raster_fallback", False))

        text_blocks = []
        for block in page.get("text_blocks") or []:
            if not isinstance(block, dict) or "id" not in block:
                continue
            block_id = str(block["id"])
            # Keep IR-native role here; callers/UI apply role_overrides for display.
            text_blocks.append(
                {
                    "id": block_id,
                    "kind": "text",
                    "bbox": _as_bbox(block.get("bbox")),
                    "text": str(block.get("text") or ""),
                    "block_role": str(block.get("block_role") or "body"),
                    "reading_order": int(block.get("reading_order") or 0),
                    "excluded": block_id in excluded,
                }
            )

        tables = []
        for table in page.get("tables") or []:
            if not isinstance(table, dict) or "id" not in table:
                continue
            table_id = str(table["id"])
            tables.append(
                {
                    "id": table_id,
                    "kind": "table",
                    "bbox": _as_bbox(table.get("bbox")),
                    "row_count": int(table.get("row_count") or 0),
                    "col_count": int(table.get("col_count") or 0),
                    "cells": _preview_table_cells(table),
                    "excluded": table_id in excluded,
                }
            )

        images = []
        for image in page.get("images") or []:
            if not isinstance(image, dict) or "id" not in image:
                continue
            image_id = str(image["id"])
            images.append(
                {
                    "id": image_id,
                    "kind": "image",
                    "bbox": _as_bbox(image.get("bbox")),
                    "excluded": image_id in excluded,
                }
            )

        shapes = []
        for shape in page.get("shapes") or []:
            if not isinstance(shape, dict) or "id" not in shape:
                continue
            shape_id = str(shape["id"])
            shapes.append(
                {
                    "id": shape_id,
                    "kind": "shape",
                    "shape_kind": str(shape.get("kind") or "rect"),
                    "bbox": _as_bbox(shape.get("bbox")),
                    "excluded": shape_id in excluded,
                }
            )

        background_mode = str(page.get("background_mode") or "elements")
        # Overlay/full-page rasters bake photos and many shapes into the background
        # image; excluding IR shapes/images does not change the baked pixels.
        baked_background = background_mode in {"overlay", "full-page"}
        pages_out.append(
            {
                "page": page_number,
                "image_url": f"/jobs/{job_id}/pages/{page_number}.jpg",
                "thumbnail_url": (
                    f"/jobs/{job_id}/pages/{page_number}.jpg?width={THUMBNAIL_WIDTH_PX}"
                ),
                "width_pt": float(page["width_pt"]),
                "height_pt": float(page["height_pt"]),
                "page_kind": str(page.get("page_kind") or "digital"),
                "background_mode": background_mode,
                "baked_background": baked_background,
                "raster_fallback": raster_fallback,
                "text_blocks": text_blocks,
                "tables": tables,
                "images": images,
                "shapes": shapes,
            }
        )

    return {
        "job_id": job_id,
        "page_count": len(pages_out),
        "slide_width_pt": float(slideir["slide_width_pt"]),
        "slide_height_pt": float(slideir["slide_height_pt"]),
        "pages": pages_out,
        "edits": edits_norm,
    }


def apply_preview_edits_to_slideir(
    slideir: dict[str, Any],
    edits: dict[str, Any],
    *,
    ensure_full_page_background: dict[int, bytes] | None = None,
) -> dict[str, Any]:
    """Return a deep-copied SlideIR with preview edits applied for ``from-ir``.

    ``ensure_full_page_background`` maps page_number → PNG/JPEG bytes used when
    a page is marked raster-fallback but has no background asset yet. Callers
    that pass bytes must also write those assets under assets_dir before
    ``from-ir`` (see :func:`materialize_raster_fallback_backgrounds`).
    """
    issues = validate_slide_ir_payload(slideir)
    if issues:
        raise SlideIrError("Invalid SlideIR: " + "; ".join(issues[:5]))

    edits_norm = normalize_preview_edits(edits)
    edits_by_page = {int(page["page"]): page for page in edits_norm["pages"]}
    result = copy.deepcopy(slideir)
    backgrounds = ensure_full_page_background or {}

    for page in result.get("pages") or []:
        if not isinstance(page, dict):
            continue
        page_number = int(page["page_number"])
        page_edits = edits_by_page.get(page_number)
        if page_edits is None:
            continue

        role_overrides = dict(page_edits.get("role_overrides") or {})
        excluded = set(page_edits.get("excluded_ids") or [])

        for block in page.get("text_blocks") or []:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("id") or "")
            if block_id in role_overrides:
                role = role_overrides[block_id]
                block["block_role"] = role
                block["use_title_placeholder"] = role == "title"

        for array_name in ELEMENT_KIND_TO_ARRAY.values():
            items = page.get(array_name)
            if not isinstance(items, list):
                continue
            page[array_name] = [
                item
                for item in items
                if not (isinstance(item, dict) and str(item.get("id") or "") in excluded)
            ]

        if page.get("reading_order"):
            page["reading_order"] = [
                block_id
                for block_id in page["reading_order"]
                if isinstance(block_id, str) and block_id not in excluded
            ]

        if page_edits.get("raster_fallback"):
            page["background_mode"] = "full-page"
            page["fallback_reason"] = "preview-raster-fallback"
            page["text_blocks"] = []
            page["shapes"] = []
            page["tables"] = []
            page["images"] = []
            page["groups"] = []
            page["reading_order"] = []
            if page.get("background") is None and page_number in backgrounds:
                # Placeholder path; materialize_raster_fallback_backgrounds rewrites it.
                page["background"] = {
                    "path": f"pages/{page_number:03d}/background.jpg",
                    "media_type": "image/jpeg",
                }

    return result


def materialize_raster_fallback_backgrounds(
    edited_slideir: dict[str, Any],
    *,
    assets_root: Path,
    page_images: dict[int, bytes],
) -> None:
    """Write full-page raster backgrounds for preview raster-fallback pages."""
    assets_root.mkdir(parents=True, exist_ok=True)
    for page in edited_slideir.get("pages") or []:
        if not isinstance(page, dict):
            continue
        if str(page.get("background_mode")) != "full-page":
            continue
        page_number = int(page["page_number"])
        image_bytes = page_images.get(page_number)
        if not image_bytes:
            continue
        page_dir = assets_root / "pages" / f"{page_number:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        filename = "background.jpg"
        (page_dir / filename).write_bytes(image_bytes)
        page["background"] = {
            "path": f"pages/{page_number:03d}/{filename}",
            "media_type": "image/jpeg",
        }


def _as_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]


def _preview_table_cells(table: dict[str, Any]) -> list[dict[str, Any]]:
    """Approximate per-cell bboxes from table column/row sizes for overlay drawing."""
    bbox = _as_bbox(table.get("bbox"))
    x0, y0, x1, y1 = bbox
    widths = [float(w) for w in (table.get("column_widths") or []) if isinstance(w, (int, float))]
    heights = [float(h) for h in (table.get("row_heights") or []) if isinstance(h, (int, float))]
    cells_out: list[dict[str, Any]] = []

    if not widths or not heights or sum(widths) <= 0 or sum(heights) <= 0:
        for cell in table.get("cells") or []:
            if not isinstance(cell, dict):
                continue
            cells_out.append(
                {
                    "row": int(cell.get("row") or 0),
                    "col": int(cell.get("col") or 0),
                    "text": str(cell.get("text") or ""),
                    "bbox": bbox,
                }
            )
        return cells_out

    scale_x = (x1 - x0) / sum(widths)
    scale_y = (y1 - y0) / sum(heights)
    col_edges = [x0]
    for width in widths:
        col_edges.append(col_edges[-1] + width * scale_x)
    row_edges = [y0]
    for height in heights:
        row_edges.append(row_edges[-1] + height * scale_y)

    for cell in table.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        row = int(cell.get("row") or 0)
        col = int(cell.get("col") or 0)
        rowspan = max(1, int(cell.get("rowspan") or 1))
        colspan = max(1, int(cell.get("colspan") or 1))
        if row < 0 or col < 0 or row >= len(heights) or col >= len(widths):
            cell_bbox = bbox
        else:
            r1 = min(row + rowspan, len(heights))
            c1 = min(col + colspan, len(widths))
            cell_bbox = [col_edges[col], row_edges[row], col_edges[c1], row_edges[r1]]
        cells_out.append(
            {
                "row": row,
                "col": col,
                "text": str(cell.get("text") or ""),
                "bbox": cell_bbox,
            }
        )
    return cells_out
