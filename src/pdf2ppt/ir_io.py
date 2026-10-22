"""SlideIR serialize / deserialize for agent round-trips.

SlideIR is a versioned JSON deck representation covering what P1/P2 already
build in memory (text, shapes, tables, pictures, layout regions, background).
Raster assets are stored as files under ``{ir_filename}.assets/`` next to the
IR JSON (e.g. ``deck.slideir.json.assets/``); paths inside the JSON are
relative to that assets directory.

This module is MIT-original; it does not copy AGPL third-party code.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .core import InputValidationError
from .models import (
    GroupElement,
    ImagePlacement,
    LayoutRegion,
    PageResult,
    QualityScore,
    ShapeElement,
    TableCell,
    TableElement,
    TextBlock,
    TextParagraph,
)

logger = logging.getLogger(__name__)

SLIDEIR_SCHEMA_VERSION = "1.0"
SLIDEIR_SCHEMA_RESOURCE = "slideir.schema.json"
DEFAULT_ASSETS_DIR_SUFFIX = ".assets"

# Soft advisory only — relative-path assets have no hard size cap. Kept for
# docs / future embedded-base64 support if agents need a single-file IR.
MAX_EMBEDDED_ASSET_BYTES = 2 * 1024 * 1024


class SlideIrError(InputValidationError):
    """Invalid or unreadable SlideIR payload."""

    error_code = "slideir-error"


@dataclass(slots=True)
class SlideIrDeck:
    """In-memory SlideIR deck ready for ``ppt_render``."""

    pages: list[PageResult]
    slide_width_pt: float
    slide_height_pt: float
    schema_version: str = SLIDEIR_SCHEMA_VERSION
    pdf2ppt_version: str | None = None
    source_path: str | None = None
    assets_dir_name: str = ""
    plan_metadata: dict[str, Any] = field(default_factory=dict)


def slideir_schema_path() -> Path:
    """Return a filesystem path to the bundled JSON Schema when available.

    Traversable package resources that are not real files (zip imports) raise
    ``SlideIrError``; callers can still use ``load_slideir_schema()``.
    """
    package = resources.files("pdf2ppt.schemas")
    resource = package.joinpath(SLIDEIR_SCHEMA_RESOURCE)
    try:
        with resources.as_file(resource) as path:
            resolved = Path(path)
            if not resolved.is_file():
                raise SlideIrError("Bundled SlideIR schema file is missing.")
            # Copy into a stable location only when as_file gave a temp extract
            # that disappears after the context; for normal installs the path
            # remains valid. Prefer reading via load_slideir_schema() in code.
            return resolved
    except (FileNotFoundError, OSError, TypeError) as error:
        raise SlideIrError(f"Unable to resolve SlideIR schema path: {error}") from error


def load_slideir_schema() -> dict[str, Any]:
    """Load the SlideIR JSON Schema document as a dict."""
    package = resources.files("pdf2ppt.schemas")
    text = package.joinpath(SLIDEIR_SCHEMA_RESOURCE).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SlideIrError("Bundled SlideIR schema is not a JSON object.")
    return payload


def default_assets_dir_name(ir_path: Path) -> str:
    """Convention: ``deck.slideir.json`` → ``deck.slideir.json.assets``."""
    return f"{ir_path.name}{DEFAULT_ASSETS_DIR_SUFFIX}"


def write_slide_ir(
    pages: list[PageResult],
    ir_path: Path,
    *,
    slide_width_pt: float | None = None,
    slide_height_pt: float | None = None,
    source_path: str | Path | None = None,
    pdf2ppt_version: str | None = None,
    plan_metadata: dict[str, Any] | None = None,
    assets_dir_name: str | None = None,
) -> Path:
    """Serialize analyzed pages to SlideIR JSON + sibling assets directory.

    Returns the written IR path.
    """
    if not pages:
        raise SlideIrError("Cannot emit SlideIR with zero pages.")

    ir_path = Path(ir_path)
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    assets_name = assets_dir_name or default_assets_dir_name(ir_path)
    assets_root = ir_path.parent / assets_name
    if assets_root.exists():
        shutil.rmtree(assets_root)
    assets_root.mkdir(parents=True, exist_ok=True)

    width = float(slide_width_pt if slide_width_pt is not None else pages[0].width_pt)
    height = float(slide_height_pt if slide_height_pt is not None else pages[0].height_pt)

    page_payloads: list[dict[str, Any]] = []
    for page in pages:
        page_payloads.append(_serialize_page(page, assets_root=assets_root))

    deck: dict[str, Any] = {
        "schema_version": SLIDEIR_SCHEMA_VERSION,
        "pdf2ppt_version": pdf2ppt_version,
        "source": {
            "path": str(source_path) if source_path is not None else None,
            "page_count": len(pages),
        },
        "assets_dir": assets_name,
        "slide_width_pt": width,
        "slide_height_pt": height,
        "pages": page_payloads,
    }
    if plan_metadata:
        deck["plan_metadata"] = plan_metadata

    ir_path.write_text(json.dumps(deck, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote SlideIR (%s page(s)) to %s (assets: %s)", len(pages), ir_path, assets_root)
    return ir_path


def read_slide_ir(ir_path: Path) -> SlideIrDeck:
    """Load SlideIR JSON and resolve asset files into ``PageResult`` objects."""
    ir_path = Path(ir_path)
    if not ir_path.exists():
        raise SlideIrError(f"SlideIR file does not exist: {ir_path}")
    try:
        raw = json.loads(ir_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SlideIrError(f"SlideIR is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise SlideIrError("SlideIR root must be a JSON object.")
    return deserialize_slide_ir(raw, ir_path=ir_path)


def deserialize_slide_ir(payload: dict[str, Any], *, ir_path: Path | None = None) -> SlideIrDeck:
    """Deserialize a SlideIR dict; ``ir_path`` is required to resolve relative assets."""
    version = str(payload.get("schema_version") or "")
    if version != SLIDEIR_SCHEMA_VERSION:
        raise SlideIrError(
            f"Unsupported SlideIR schema_version {version!r}; expected {SLIDEIR_SCHEMA_VERSION!r}."
        )

    pages_raw = payload.get("pages")
    if not isinstance(pages_raw, list) or not pages_raw:
        raise SlideIrError("SlideIR must contain a non-empty 'pages' array.")

    try:
        slide_width_pt = float(payload["slide_width_pt"])
        slide_height_pt = float(payload["slide_height_pt"])
    except (KeyError, TypeError, ValueError) as error:
        raise SlideIrError("SlideIR requires numeric slide_width_pt and slide_height_pt.") from error
    if slide_width_pt <= 0 or slide_height_pt <= 0:
        raise SlideIrError("slide_width_pt and slide_height_pt must be positive.")

    assets_dir_name = str(payload.get("assets_dir") or "")
    assets_root: Path | None = None
    if ir_path is not None:
        if not assets_dir_name:
            assets_dir_name = default_assets_dir_name(ir_path)
        assets_root = (ir_path.parent / assets_dir_name).resolve()
        _assert_path_inside(assets_root, ir_path.parent.resolve())

    pages: list[PageResult] = []
    for index, page_payload in enumerate(pages_raw):
        if not isinstance(page_payload, dict):
            raise SlideIrError(f"pages[{index}] must be an object.")
        pages.append(_deserialize_page(page_payload, assets_root=assets_root, page_index=index))

    source_raw = payload.get("source")
    source: dict[str, Any] = source_raw if isinstance(source_raw, dict) else {}
    plan_raw = payload.get("plan_metadata")
    plan_metadata: dict[str, Any] = dict(plan_raw) if isinstance(plan_raw, dict) else {}

    return SlideIrDeck(
        pages=pages,
        slide_width_pt=slide_width_pt,
        slide_height_pt=slide_height_pt,
        schema_version=version,
        pdf2ppt_version=str(payload["pdf2ppt_version"]) if payload.get("pdf2ppt_version") else None,
        source_path=str(source["path"]) if source.get("path") else None,
        assets_dir_name=assets_dir_name,
        plan_metadata=plan_metadata,
    )


def validate_slide_ir_payload(payload: dict[str, Any]) -> list[str]:
    """Lightweight structural validation (no jsonschema dependency).

    Returns a list of human-readable issues (empty = ok). Full Draft-2020-12
    checking is available via the bundled schema file for external tools.
    """
    issues: list[str] = []
    if payload.get("schema_version") != SLIDEIR_SCHEMA_VERSION:
        issues.append(f"schema_version must be {SLIDEIR_SCHEMA_VERSION!r}")
    if not isinstance(payload.get("pages"), list) or not payload["pages"]:
        issues.append("pages must be a non-empty array")
    for key in ("slide_width_pt", "slide_height_pt"):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            issues.append(f"{key} must be a positive number")
    for index, page in enumerate(payload.get("pages") or []):
        if not isinstance(page, dict):
            issues.append(f"pages[{index}] must be an object")
            continue
        for required in ("page_number", "page_kind", "background_mode", "width_pt", "height_pt", "text_blocks"):
            if required not in page:
                issues.append(f"pages[{index}] missing required field {required!r}")
        if not isinstance(page.get("text_blocks"), list):
            issues.append(f"pages[{index}].text_blocks must be an array")
    return issues


def render_deck_to_pptx(deck: SlideIrDeck, output_path: Path) -> Path:
    """Render a deserialized SlideIR deck to a PPTX file."""
    from pptx import Presentation

    from .ppt_render import pt_to_emu, render_page_to_slide

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    presentation = Presentation()
    presentation.slide_width = pt_to_emu(deck.slide_width_pt)
    presentation.slide_height = pt_to_emu(deck.slide_height_pt)
    blank_layout = presentation.slide_layouts[6]
    for page in deck.pages:
        render_page_to_slide(
            presentation,
            blank_layout,
            page,
            slide_width_pt=deck.slide_width_pt,
            slide_height_pt=deck.slide_height_pt,
        )
    presentation.save(str(output_path))
    logger.info("Rendered SlideIR -> %s (%s slide(s))", output_path, len(deck.pages))
    return output_path


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_page(page: PageResult, *, assets_root: Path) -> dict[str, Any]:
    page_dir = assets_root / "pages" / f"{page.page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    rel_page = f"pages/{page.page_number:03d}"

    background_ref: dict[str, Any] | None = None
    if page.background_image_bytes:
        ext = _guess_image_extension(page.background_image_bytes)
        filename = f"background{ext}"
        (page_dir / filename).write_bytes(page.background_image_bytes)
        background_ref = {
            "path": f"{rel_page}/{filename}",
            "media_type": _media_type_for_extension(ext),
        }

    images: list[dict[str, Any]] = []
    for index, placement in enumerate(page.image_elements):
        filename = f"image_{index:03d}.png"
        (page_dir / filename).write_bytes(placement.png_bytes)
        images.append(
            {
                "id": f"img_{page.page_number}_{index}",
                "bbox": list(placement.bbox),
                "asset": {"path": f"{rel_page}/{filename}", "media_type": "image/png"},
                "z_order": int(getattr(placement, "z_order", index) or index),
                "provenance": "pipeline_crop",
                "confidence": 1.0,
            }
        )

    reading_order = [block.id for block in sorted(page.text_blocks, key=lambda item: item.reading_order)]

    payload: dict[str, Any] = {
        "page_number": page.page_number,
        "page_kind": page.page_kind,
        "background_mode": page.background_mode,
        "width_pt": page.width_pt,
        "height_pt": page.height_pt,
        "text_blocks": [_serialize_text_block(block) for block in page.text_blocks],
        "shapes": [shape.to_dict() for shape in page.shape_elements],
        "tables": [_serialize_table(table) for table in page.table_elements],
        "groups": [group.to_dict() for group in page.group_elements],
        "images": images,
        "layout_regions": [region.to_dict() for region in page.layout_regions],
        "background": background_ref,
        "quality_score": page.quality_score.to_dict(),
        "fallback_reason": page.fallback_reason,
        "background_inpaint_engine": page.background_inpaint_engine,
        "background_inpaint_note": page.background_inpaint_note,
        "reading_order": reading_order,
    }
    if page.vector_mode is not None:
        payload["vector_mode"] = page.vector_mode
    if page.vector_fallback_reason is not None:
        payload["vector_fallback_reason"] = page.vector_fallback_reason
    if page.layout_engine is not None:
        payload["layout_engine"] = page.layout_engine
    if page.layout_fallback_reason is not None:
        payload["layout_fallback_reason"] = page.layout_fallback_reason
    if page.plan_metadata:
        payload["plan_metadata"] = dict(page.plan_metadata)
    return payload


def _serialize_text_block(block: TextBlock) -> dict[str, Any]:
    payload: dict[str, Any] = dict(block.to_dict())
    # Agents treat text as ground truth; mark native/OCR evidence as locked.
    payload.setdefault("locked", True)
    payload.setdefault("provenance", f"text_{block.source}")
    polygon = payload.get("image_polygon")
    if isinstance(polygon, (list, tuple)):
        payload["image_polygon"] = [list(point) for point in polygon]
    image_bbox = payload.get("image_bbox")
    if isinstance(image_bbox, (list, tuple)):
        payload["image_bbox"] = list(image_bbox)
    bbox = payload.get("bbox")
    if isinstance(bbox, (list, tuple)):
        payload["bbox"] = list(bbox)
    line_bboxes = payload.get("line_bboxes")
    if isinstance(line_bboxes, list):
        payload["line_bboxes"] = [
            list(box) if isinstance(box, (list, tuple)) else box for box in line_bboxes
        ]
    return payload


def _serialize_table(table: TableElement) -> dict[str, Any]:
    payload: dict[str, Any] = dict(table.to_dict())
    bbox = payload.get("bbox")
    if isinstance(bbox, (list, tuple)):
        payload["bbox"] = list(bbox)
    return payload


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------


def _deserialize_page(
    payload: dict[str, Any],
    *,
    assets_root: Path | None,
    page_index: int,
) -> PageResult:
    try:
        page_number = int(payload["page_number"])
        page_kind = str(payload["page_kind"])
        background_mode = str(payload["background_mode"])
        width_pt = float(payload["width_pt"])
        height_pt = float(payload["height_pt"])
    except (KeyError, TypeError, ValueError) as error:
        raise SlideIrError(f"pages[{page_index}] has invalid required geometry fields.") from error

    if page_kind not in {"digital", "scanned", "hybrid"}:
        raise SlideIrError(f"pages[{page_index}].page_kind is invalid: {page_kind!r}")
    if background_mode not in {"elements", "overlay", "full-page"}:
        raise SlideIrError(f"pages[{page_index}].background_mode is invalid: {background_mode!r}")

    text_blocks = [
        _deserialize_text_block(item, page_index=page_index, block_index=block_index)
        for block_index, item in enumerate(payload.get("text_blocks") or [])
    ]
    shape_elements = [
        _deserialize_shape(item, page_index=page_index, shape_index=shape_index)
        for shape_index, item in enumerate(payload.get("shapes") or [])
    ]
    table_elements = [
        _deserialize_table(item, page_index=page_index, table_index=table_index)
        for table_index, item in enumerate(payload.get("tables") or [])
    ]
    group_elements = [
        _deserialize_group(item, page_index=page_index, group_index=group_index)
        for group_index, item in enumerate(payload.get("groups") or [])
    ]
    layout_regions = [
        _deserialize_layout_region(item, page_index=page_index, region_index=region_index)
        for region_index, item in enumerate(payload.get("layout_regions") or [])
    ]

    image_elements: list[ImagePlacement] = []
    for image_index, item in enumerate(payload.get("images") or []):
        if not isinstance(item, dict):
            raise SlideIrError(f"pages[{page_index}].images[{image_index}] must be an object.")
        bbox = _require_bbox(item.get("bbox"), label=f"pages[{page_index}].images[{image_index}].bbox")
        asset = item.get("asset")
        if not isinstance(asset, dict) or not asset.get("path"):
            raise SlideIrError(f"pages[{page_index}].images[{image_index}] needs asset.path")
        png_bytes = _load_asset_bytes(asset["path"], assets_root=assets_root)
        image_elements.append(
            ImagePlacement(
                bbox=bbox,
                png_bytes=png_bytes,
                z_order=int(item.get("z_order", image_index) or image_index),
            )
        )

    background_bytes: bytes | None = None
    background = payload.get("background")
    if background is not None:
        if not isinstance(background, dict) or not background.get("path"):
            raise SlideIrError(f"pages[{page_index}].background must be null or an asset ref with path.")
        background_bytes = _load_asset_bytes(background["path"], assets_root=assets_root)

    quality_value = payload.get("quality_score")
    quality_raw: dict[str, Any] = quality_value if isinstance(quality_value, dict) else {}
    quality_score = QualityScore(
        text_confidence=float(quality_raw.get("text_confidence", 0.0)),
        layout_overlap_score=float(quality_raw.get("layout_overlap_score", 0.0)),
        style_recovery_score=float(quality_raw.get("style_recovery_score", 0.0)),
        editable_ratio=float(quality_raw.get("editable_ratio", 0.0)),
    )

    plan_raw = payload.get("plan_metadata")
    plan_metadata: dict[str, object] = dict(plan_raw) if isinstance(plan_raw, dict) else {}

    return PageResult(
        page_number=page_number,
        page_kind=page_kind,  # type: ignore[arg-type]
        background_mode=background_mode,  # type: ignore[arg-type]
        width_pt=width_pt,
        height_pt=height_pt,
        text_blocks=text_blocks,
        quality_score=quality_score,
        fallback_reason=payload.get("fallback_reason"),
        background_inpaint_engine=payload.get("background_inpaint_engine"),
        background_inpaint_note=payload.get("background_inpaint_note"),
        background_image_bytes=background_bytes,
        image_elements=image_elements,
        shape_elements=shape_elements,
        table_elements=table_elements,
        group_elements=group_elements,
        layout_regions=layout_regions,
        vector_mode=payload.get("vector_mode"),
        vector_fallback_reason=payload.get("vector_fallback_reason"),
        layout_engine=payload.get("layout_engine"),
        layout_fallback_reason=payload.get("layout_fallback_reason"),
        plan_metadata=plan_metadata,
    )


def _deserialize_text_block(payload: Any, *, page_index: int, block_index: int) -> TextBlock:
    if not isinstance(payload, dict):
        raise SlideIrError(f"pages[{page_index}].text_blocks[{block_index}] must be an object.")
    try:
        source = str(payload["source"])
        if source not in {"native", "ocr"}:
            raise SlideIrError(f"invalid text source {source!r}")
        paragraphs_raw = payload.get("paragraphs")
        paragraphs: list[TextParagraph] | None = None
        if paragraphs_raw is not None:
            if not isinstance(paragraphs_raw, list):
                raise SlideIrError("paragraphs must be an array")
            paragraphs = [_deserialize_paragraph(item) for item in paragraphs_raw]
        image_polygon = None
        if payload.get("image_polygon") is not None:
            image_polygon = tuple(
                (float(point[0]), float(point[1])) for point in payload["image_polygon"]
            )
        image_bbox = None
        if payload.get("image_bbox") is not None:
            image_bbox = _require_bbox(payload["image_bbox"], label="image_bbox")
        line_bboxes = None
        if payload.get("line_bboxes") is not None:
            raw_lines = payload["line_bboxes"]
            if not isinstance(raw_lines, list):
                raise SlideIrError("line_bboxes must be an array")
            line_bboxes = [
                _require_bbox(item, label=f"line_bboxes[{index}]")
                for index, item in enumerate(raw_lines)
            ]
        return TextBlock(
            id=str(payload["id"]),
            source=source,  # type: ignore[arg-type]
            bbox=_require_bbox(payload["bbox"], label="bbox"),
            text=str(payload.get("text") or ""),
            confidence=float(payload.get("confidence", 0.0)),
            font_family=payload.get("font_family"),
            font_size=_optional_float(payload.get("font_size")),
            font_color=payload.get("font_color"),
            bold=bool(payload.get("bold", False)),
            italic=bool(payload.get("italic", False)),
            reading_order=int(payload.get("reading_order", 0)),
            block_role=str(payload.get("block_role") or "body"),
            image_bbox=image_bbox,
            image_polygon=image_polygon,
            layout_region_id=payload.get("layout_region_id"),
            layout_label=payload.get("layout_label"),
            paragraphs=paragraphs,
            line_bboxes=line_bboxes,
            use_title_placeholder=bool(payload.get("use_title_placeholder", False)),
            pin_font_size=bool(payload.get("pin_font_size", False)),
        )
    except (KeyError, TypeError, ValueError, SlideIrError) as error:
        raise SlideIrError(
            f"pages[{page_index}].text_blocks[{block_index}] is invalid: {error}"
        ) from error


def _deserialize_paragraph(payload: Any) -> TextParagraph:
    if not isinstance(payload, dict):
        raise SlideIrError("paragraph must be an object")
    return TextParagraph(
        text=str(payload.get("text") or ""),
        list_level=_optional_int(payload.get("list_level")),
        list_style=payload.get("list_style"),
        bold=payload.get("bold"),
        italic=payload.get("italic"),
        font_size=_optional_float(payload.get("font_size")),
        font_color=payload.get("font_color"),
        space_before_pt=_optional_float(payload.get("space_before_pt")),
        text_indent_pt=_optional_float(payload.get("text_indent_pt")),
        left_pt=_optional_float(payload.get("left_pt")),
        baseline_pt=_optional_float(payload.get("baseline_pt")),
    )


def _deserialize_shape(payload: Any, *, page_index: int, shape_index: int) -> ShapeElement:
    if not isinstance(payload, dict):
        raise SlideIrError(f"pages[{page_index}].shapes[{shape_index}] must be an object.")
    try:
        points_raw = payload.get("points") or []
        points = [(float(point[0]), float(point[1])) for point in points_raw]
        return ShapeElement(
            id=str(payload["id"]),
            kind=str(payload["kind"]),  # type: ignore[arg-type]
            bbox=_require_bbox(payload["bbox"], label="bbox"),
            points=points,
            fill_color=payload.get("fill_color"),
            stroke_color=payload.get("stroke_color"),
            stroke_width=_optional_float(payload.get("stroke_width")),
            stroke_dash=payload.get("stroke_dash"),
            fill_opacity=float(payload.get("fill_opacity", 1.0)),
            stroke_opacity=float(payload.get("stroke_opacity", 1.0)),
            z_order=int(payload.get("z_order", 0)),
            provenance=str(payload.get("provenance") or "native_drawing"),
            confidence=float(payload.get("confidence", 1.0)),
            locked=bool(payload.get("locked", True)),
            drawing_index=_optional_int(payload.get("drawing_index")),
            is_page_background=bool(payload.get("is_page_background", False)),
            render_under_raster=bool(payload.get("render_under_raster", False)),
            render_above_tables=bool(payload.get("render_above_tables", False)),
            arrow_direction=payload.get("arrow_direction"),
        )
    except (KeyError, TypeError, ValueError, SlideIrError) as error:
        raise SlideIrError(f"pages[{page_index}].shapes[{shape_index}] is invalid: {error}") from error


def _deserialize_table(payload: Any, *, page_index: int, table_index: int) -> TableElement:
    if not isinstance(payload, dict):
        raise SlideIrError(f"pages[{page_index}].tables[{table_index}] must be an object.")
    try:
        cells = [
            TableCell(
                row=int(cell["row"]),
                col=int(cell["col"]),
                text=str(cell.get("text") or ""),
                rowspan=int(cell.get("rowspan", 1)),
                colspan=int(cell.get("colspan", 1)),
                fill_color=cell.get("fill_color"),
                font_size=_optional_float(cell.get("font_size")),
                font_color=cell.get("font_color"),
                font_name=cell.get("font_name"),
                bold=bool(cell.get("bold", False)),
                horizontal_align=str(cell.get("horizontal_align") or "left"),
                vertical_anchor=str(cell.get("vertical_anchor") or "middle"),
                text_block_ids=[str(item) for item in (cell.get("text_block_ids") or [])],
                margin_left=_optional_float(cell.get("margin_left")),
                margin_right=_optional_float(cell.get("margin_right")),
                margin_top=_optional_float(cell.get("margin_top")),
            )
            for cell in (payload.get("cells") or [])
        ]
        return TableElement(
            id=str(payload["id"]),
            bbox=_require_bbox(payload["bbox"], label="bbox"),
            row_count=int(payload["row_count"]),
            col_count=int(payload["col_count"]),
            cells=cells,
            column_widths=[float(value) for value in payload.get("column_widths") or []],
            row_heights=[float(value) for value in payload.get("row_heights") or []],
            border_color=payload.get("border_color"),
            border_width=_optional_float(payload.get("border_width")),
            outer_border_width=_optional_float(payload.get("outer_border_width")),
            inner_border_width=_optional_float(payload.get("inner_border_width")),
            # Missing z_order must still sit above typical shapes (main always
            # drew tables last). Default 10000+index matches the PDF pipeline.
            z_order=int(payload["z_order"]) if payload.get("z_order") is not None else 10_000 + table_index,
            provenance=str(payload.get("provenance") or "native_table"),
            confidence=float(payload.get("confidence", 0.9)),
            locked=bool(payload.get("locked", True)),
        )
    except (KeyError, TypeError, ValueError, SlideIrError) as error:
        raise SlideIrError(f"pages[{page_index}].tables[{table_index}] is invalid: {error}") from error


def _deserialize_group(payload: Any, *, page_index: int, group_index: int) -> GroupElement:
    if not isinstance(payload, dict):
        raise SlideIrError(f"pages[{page_index}].groups[{group_index}] must be an object.")
    bbox = None
    if payload.get("bbox") is not None:
        bbox = _require_bbox(payload["bbox"], label="bbox")
    return GroupElement(
        id=str(payload["id"]),
        member_ids=[str(item) for item in (payload.get("member_ids") or [])],
        bbox=bbox,
        provenance=str(payload.get("provenance") or "native_group"),
        confidence=float(payload.get("confidence", 0.5)),
        locked=bool(payload.get("locked", False)),
    )


def _deserialize_layout_region(payload: Any, *, page_index: int, region_index: int) -> LayoutRegion:
    if not isinstance(payload, dict):
        raise SlideIrError(f"pages[{page_index}].layout_regions[{region_index}] must be an object.")
    return LayoutRegion(
        id=str(payload["id"]),
        label=str(payload["label"]),  # type: ignore[arg-type]
        bbox=_require_bbox(payload["bbox"], label="bbox"),
        confidence=float(payload.get("confidence", 1.0)),
        reading_order=int(payload.get("reading_order", 0)),
        raw_label=str(payload.get("raw_label") or ""),
        engine=str(payload.get("engine") or "heuristic"),
    )


# ---------------------------------------------------------------------------
# Asset / path helpers
# ---------------------------------------------------------------------------


def _load_asset_bytes(relative_path: str, *, assets_root: Path | None) -> bytes:
    if assets_root is None:
        raise SlideIrError(
            f"Cannot resolve asset {relative_path!r} without an IR file path "
            "(assets live under assets_dir next to the SlideIR JSON)."
        )
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SlideIrError(f"Asset path must be relative and stay under assets_dir: {relative_path!r}")
    resolved = (assets_root / relative).resolve()
    _assert_path_inside(resolved, assets_root)
    if not resolved.exists():
        raise SlideIrError(f"Missing SlideIR asset: {resolved}")
    return resolved.read_bytes()


def _assert_path_inside(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SlideIrError(f"Path escapes assets root: {path}") from error


def _require_bbox(value: Any, *, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SlideIrError(f"{label} must be a 4-number array")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _guess_image_extension(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _media_type_for_extension(ext: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext.lower(), "application/octet-stream")
