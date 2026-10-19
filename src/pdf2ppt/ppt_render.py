from __future__ import annotations

import io
import logging
import math
from typing import Any

from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, MSO_VERTICAL_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Pt

from .models import ShapeElement, TableElement, TextBlock, TextParagraph
from .text_style import (
    choose_measurement_font,
    classify_text_script,
    default_font_family,
    default_ppt_typefaces,
    measure_text_dimensions,
    ocr_fit_height_cap_ratio,
    single_line_fit_width_ratio,
)

logger = logging.getLogger(__name__)

EMU_PER_PT = 12700

_SHAPE_KIND_TO_MSO = {
    "rect": MSO_SHAPE.RECTANGLE,
    "round_rect": MSO_SHAPE.ROUNDED_RECTANGLE,
    "ellipse": MSO_SHAPE.OVAL,
    "arrow": MSO_SHAPE.RIGHT_ARROW,
}

_ARROW_DIRECTION_TO_MSO = {
    "left": MSO_SHAPE.LEFT_ARROW,
    "right": MSO_SHAPE.RIGHT_ARROW,
    "up": MSO_SHAPE.UP_ARROW,
    "down": MSO_SHAPE.DOWN_ARROW,
}

# ECMA-376 CT_TableCellProperties child order before fill.
_TCPR_BORDER_TAGS = ("a:lnL", "a:lnR", "a:lnT", "a:lnB", "a:lnTlToBr", "a:lnBlToTr", "a:cell3D")
_TCPR_FILL_TAGS = ("a:solidFill", "a:noFill", "a:gradFill", "a:blipFill", "a:pattFill", "a:grpFill")


def render_page_to_slide(
    presentation: Any,
    blank_layout: Any,
    page_result: Any,
    *,
    slide_width_pt: float,
    slide_height_pt: float,
) -> None:
    title_block = _select_title_placeholder_block(page_result.text_blocks)
    slide_layout = blank_layout
    if title_block is not None:
        title_layout = _find_title_only_layout(presentation)
        if title_layout is not None:
            slide_layout = title_layout
        else:
            title_block = None

    slide = presentation.slides.add_slide(slide_layout)
    scale_x = slide_width_pt / page_result.width_pt
    scale_y = slide_height_pt / page_result.height_pt

    background_element: Any | None = None
    if page_result.background_image_bytes is not None:
        background_picture = slide.shapes.add_picture(
            io.BytesIO(page_result.background_image_bytes),
            0,
            0,
            width=presentation.slide_width,
            height=presentation.slide_height,
        )
        background_element = background_picture._element  # noqa: SLF001

    # Z-order: page-background shapes → other shapes → pictures → tables → text.
    shape_elements = list(getattr(page_result, "shape_elements", None) or [])
    background_shapes = [shape for shape in shape_elements if shape.is_page_background]
    foreground_shapes = [shape for shape in shape_elements if not shape.is_page_background]
    for shape in sorted(background_shapes, key=lambda item: item.z_order):
        added = add_shape_element(slide, shape, scale_x=scale_x, scale_y=scale_y)
        if background_element is None and added is not None:
            background_element = getattr(added, "_element", None)
    for shape in sorted(foreground_shapes, key=lambda item: item.z_order):
        add_shape_element(slide, shape, scale_x=scale_x, scale_y=scale_y)

    for image_element in page_result.image_elements:
        left, top, width, height = bbox_to_shape_geometry(image_element.bbox, scale_x, scale_y)
        slide.shapes.add_picture(
            io.BytesIO(image_element.png_bytes),
            left,
            top,
            width=width,
            height=height,
        )

    for table in sorted(getattr(page_result, "table_elements", None) or [], key=lambda item: item.z_order):
        add_table_element(slide, table, scale_x=scale_x, scale_y=scale_y)

    if title_block is not None and slide.shapes.title is not None:
        _apply_title_placeholder(
            slide.shapes.title,
            title_block,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        # Title Only layouts put the title placeholder first in spTree, under any
        # full-slide background picture/shape. Move it just after the background.
        raise_title_above_background(slide, background_element=background_element)

    for block in sorted(page_result.text_blocks, key=lambda item: item.reading_order):
        if title_block is not None and block.id == title_block.id:
            continue
        add_text_block(slide, block, scale_x=scale_x, scale_y=scale_y)


def raise_title_above_background(slide: Any, *, background_element: Any | None) -> None:
    """Move the title placeholder immediately after the slide background element."""

    title_shape = getattr(slide.shapes, "title", None)
    if title_shape is None:
        return
    try:
        title_element = title_shape._element  # noqa: SLF001
        tree = title_element.getparent()
        if tree is None:
            return
        tree.remove(title_element)
        if background_element is not None and background_element.getparent() is tree:
            background_element.addnext(title_element)
        else:
            # No dedicated background: place title after any leading non-title shapes
            # that already sit at the bottom of the tree, else append.
            tree.insert(0, title_element)
            if len(tree):
                first = tree[0]
                if first is not title_element:
                    tree.remove(title_element)
                    first.addnext(title_element)
    except Exception:  # noqa: BLE001
        logger.debug("Could not raise title above background", exc_info=True)


def _find_title_only_layout(presentation: Any) -> Any | None:
    """Return the Title Only layout when present (title placeholder, no body).

    Broader layouts (Title Slide, Section Header, Title and Content) add empty
    body placeholders that hurt fidelity and confuse shape-index assumptions,
    so they are intentionally not used.
    """

    layouts = list(getattr(presentation, "slide_layouts", []) or [])
    for layout in layouts:
        name = str(getattr(layout, "name", "") or "").strip().lower()
        if name == "title only":
            return layout
    return None


def _select_title_placeholder_block(text_blocks: list[TextBlock]) -> TextBlock | None:
    for block in text_blocks:
        if getattr(block, "use_title_placeholder", False):
            return block
    return None


def _apply_title_placeholder(title_shape: Any, block: TextBlock, *, scale_x: float, scale_y: float) -> None:
    """Fill the title placeholder and move it onto the PDF title bbox (keeps SSIM stable)."""

    left, top, width, height = bbox_to_shape_geometry(block.bbox, scale_x, scale_y)
    # Widen slightly so a single-line title does not wrap under Title Only margins.
    try:
        title_shape.left = left
        title_shape.top = top
        widened = max(int(width * 1.08), int(width), 1) if width else 1
        title_shape.width = max(widened, Emu(1))
        title_shape.height = max(height, Emu(1))
    except Exception:  # noqa: BLE001
        logger.debug("Could not reposition title placeholder to PDF bbox", exc_info=True)

    text_frame = title_shape.text_frame
    text_frame.clear()
    text_frame.word_wrap = False
    text_frame.margin_left = 0
    text_frame.margin_right = 0
    text_frame.margin_top = 0
    text_frame.margin_bottom = 0
    paragraph = text_frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.LEFT
    run = paragraph.add_run()
    run.text = block.text.splitlines()[0] if block.text else ""
    script = classify_text_script(block.text)
    font_family = block.font_family or default_font_family(script)
    run.font.name = font_family
    _set_run_typefaces(
        run,
        default_ppt_typefaces(script)
        if not block.font_family
        else {"latin": font_family, "ea": font_family, "cs": font_family},
    )
    if block.font_size:
        run.font.size = Pt(max(6.0, block.font_size * min(scale_x, scale_y)))
    if block.font_color:
        run.font.color.rgb = RGBColor.from_string(block.font_color.lstrip("#"))
    # Preserve PDF boldness — do not force title runs to bold.
    run.font.bold = block.bold
    run.font.italic = block.italic


def add_shape_element(slide: Any, shape: ShapeElement, *, scale_x: float, scale_y: float) -> Any | None:
    if shape.kind in {"line", "polyline"} and len(shape.points) >= 2:
        return add_line_or_polyline(slide, shape, scale_x=scale_x, scale_y=scale_y)
    if shape.kind in {"polygon", "freeform"} and len(shape.points) >= 3:
        return add_freeform_shape(slide, shape, scale_x=scale_x, scale_y=scale_y)

    if shape.kind == "arrow":
        mso_shape = _ARROW_DIRECTION_TO_MSO.get(shape.arrow_direction or "right", MSO_SHAPE.RIGHT_ARROW)
    else:
        mso_shape = _SHAPE_KIND_TO_MSO.get(shape.kind, MSO_SHAPE.RECTANGLE)
    left, top, width, height = bbox_to_shape_geometry(shape.bbox, scale_x, scale_y)
    # Lines stored as zero-thickness bboxes still need a positive size for autoshapes.
    auto_shape = slide.shapes.add_shape(mso_shape, left, top, max(width, Emu(1)), max(height, Emu(1)))
    apply_shape_style(auto_shape, shape)
    return auto_shape


def add_line_or_polyline(slide: Any, shape: ShapeElement, *, scale_x: float, scale_y: float) -> Any | None:
    points = shape.points
    if shape.kind == "line" or len(points) == 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        connector = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            pt_to_emu(x0 * scale_x),
            pt_to_emu(y0 * scale_y),
            pt_to_emu(x1 * scale_x),
            pt_to_emu(y1 * scale_y),
        )
        apply_line_style(connector, shape)
        return connector
    return add_freeform_shape(slide, shape, scale_x=scale_x, scale_y=scale_y, close=False)


def add_freeform_shape(
    slide: Any,
    shape: ShapeElement,
    *,
    scale_x: float,
    scale_y: float,
    close: bool | None = None,
) -> Any | None:
    points = shape.points
    if len(points) < 2:
        return None
    start_x, start_y = points[0]
    builder = slide.shapes.build_freeform(pt_to_emu(start_x * scale_x), pt_to_emu(start_y * scale_y))
    segments = [(pt_to_emu(x * scale_x), pt_to_emu(y * scale_y)) for x, y in points[1:]]
    should_close = shape.kind in {"polygon", "freeform"} if close is None else close
    if shape.kind == "polyline":
        should_close = False
    builder.add_line_segments(segments, close=should_close)
    freeform = builder.convert_to_shape()
    apply_shape_style(freeform, shape)
    return freeform


def add_table_element(slide: Any, table: TableElement, *, scale_x: float, scale_y: float) -> None:
    left, top, width, height = bbox_to_shape_geometry(table.bbox, scale_x, scale_y)
    table_shape = slide.shapes.add_table(
        table.row_count,
        table.col_count,
        left,
        top,
        max(width, Emu(1)),
        max(height, Emu(1)),
    )
    ppt_table = table_shape.table
    # Disable python-pptx's default Medium Style 2 – Accent 1 (white header text + banding).
    ppt_table.first_row = False
    ppt_table.horz_banding = False

    total_width = sum(table.column_widths) or 1.0
    usable_width = max(int(width), 1)
    for col_index, col_width_pt in enumerate(table.column_widths):
        ppt_table.columns[col_index].width = Emu(max(1, int(round(usable_width * (col_width_pt / total_width)))))

    total_height = sum(table.row_heights) or 1.0
    usable_height = max(int(height), 1)
    for row_index, row_height_pt in enumerate(table.row_heights):
        ppt_table.rows[row_index].height = Emu(max(1, int(round(usable_height * (row_height_pt / total_height)))))

    # Apply merges first (python-pptx requires the top-left cell).
    for cell in table.cells:
        if cell.rowspan <= 1 and cell.colspan <= 1:
            continue
        end_row = min(table.row_count - 1, cell.row + cell.rowspan - 1)
        end_col = min(table.col_count - 1, cell.col + cell.colspan - 1)
        if end_row == cell.row and end_col == cell.col:
            continue
        try:
            ppt_table.cell(cell.row, cell.col).merge(ppt_table.cell(end_row, end_col))
        except (ValueError, TypeError) as error:
            logger.debug("Skipping table merge at (%s,%s): %s", cell.row, cell.col, error)

    for cell in table.cells:
        ppt_cell = ppt_table.cell(cell.row, cell.col)
        # Borders before fills so tcPr child order matches ECMA-376.
        if table.border_color:
            _set_cell_borders(
                ppt_cell,
                table.border_color,
                table.border_width or 1.0,
                row=cell.row,
                col=cell.col,
                row_count=table.row_count,
                col_count=table.col_count,
                rowspan=cell.rowspan,
                colspan=cell.colspan,
                outer_width=table.outer_border_width,
                inner_width=table.inner_border_width,
            )
        if cell.fill_color:
            try:
                ppt_cell.fill.solid()
                ppt_cell.fill.fore_color.rgb = RGBColor.from_string(cell.fill_color.lstrip("#"))
            except (ValueError, TypeError):
                ppt_cell.fill.background()
        else:
            ppt_cell.fill.background()
        # python-pptx may append fills in an arbitrary position; normalize order.
        _normalize_tcpr_child_order(ppt_cell)

        ppt_cell.text = cell.text
        ppt_cell.vertical_anchor = _resolve_table_vertical_anchor(cell.vertical_anchor)
        font_color = cell.font_color or "#1A1A1A"
        font_size_pt = max(6.0, (cell.font_size or 12.0) * min(scale_x, scale_y))
        alignment = _resolve_table_paragraph_align(cell.horizontal_align)
        for paragraph in ppt_cell.text_frame.paragraphs:
            paragraph.alignment = alignment
            for run in paragraph.runs:
                run.font.size = Pt(font_size_pt)
                try:
                    run.font.color.rgb = RGBColor.from_string(font_color.lstrip("#"))
                except ValueError:
                    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
                run.font.bold = cell.bold
                if cell.font_name:
                    run.font.name = cell.font_name


def apply_shape_style(shape: Any, element: ShapeElement) -> None:
    if element.fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(element.fill_color.lstrip("#"))
        _set_fill_alpha(shape, element.fill_opacity)
    else:
        shape.fill.background()

    if element.stroke_color and (element.stroke_width is None or element.stroke_width > 0):
        shape.line.color.rgb = RGBColor.from_string(element.stroke_color.lstrip("#"))
        if element.stroke_width is not None:
            shape.line.width = Pt(max(0.25, element.stroke_width))
        if element.stroke_dash and element.stroke_dash != "solid":
            shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        _set_line_alpha(shape, element.stroke_opacity)
    else:
        shape.line.fill.background()
    _clear_inherited_effects(shape)


def apply_line_style(connector: Any, element: ShapeElement) -> None:
    if element.stroke_color:
        connector.line.color.rgb = RGBColor.from_string(element.stroke_color.lstrip("#"))
    if element.stroke_width is not None:
        connector.line.width = Pt(max(0.25, element.stroke_width))
    if element.stroke_dash and element.stroke_dash != "solid":
        connector.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    _set_line_alpha(connector, element.stroke_opacity)
    _clear_inherited_effects(connector)


def _set_fill_alpha(shape: Any, opacity: float) -> None:
    if opacity >= 0.999:
        return
    try:
        solid_fill = shape.fill._xPr.solidFill  # noqa: SLF001
        srgb = solid_fill.find(qn("a:srgbClr"))
        if srgb is None:
            return
        for child in list(srgb):
            if child.tag == qn("a:alpha"):
                srgb.remove(child)
        alpha = OxmlElement("a:alpha")
        alpha.set("val", str(int(round(max(0.0, min(1.0, opacity)) * 100000))))
        srgb.append(alpha)
    except Exception:  # noqa: BLE001 - OOXML structure varies by python-pptx version
        logger.debug("Could not set fill alpha", exc_info=True)


def _set_line_alpha(shape: Any, opacity: float) -> None:
    if opacity >= 0.999:
        return
    try:
        line_elem = shape.line._ln  # noqa: SLF001
        solid = line_elem.find(qn("a:solidFill"))
        if solid is None:
            return
        srgb = solid.find(qn("a:srgbClr"))
        if srgb is None:
            return
        for child in list(srgb):
            if child.tag == qn("a:alpha"):
                srgb.remove(child)
        alpha = OxmlElement("a:alpha")
        alpha.set("val", str(int(round(max(0.0, min(1.0, opacity)) * 100000))))
        srgb.append(alpha)
    except Exception:  # noqa: BLE001
        logger.debug("Could not set line alpha", exc_info=True)


def _set_cell_borders(
    cell: Any,
    color_hex: str,
    width_pt: float,
    *,
    row: int = 0,
    col: int = 0,
    row_count: int = 1,
    col_count: int = 1,
    rowspan: int = 1,
    colspan: int = 1,
    outer_width: float | None = None,
    inner_width: float | None = None,
) -> None:
    """Set cell borders via OOXML, with optional outer vs inner rule widths.

    ECMA-376 ``CT_TableCellProperties`` requires borders (lnL…lnB, diagonals,
    cell3D) **before** any fill element. Inserting after an existing solidFill
    can make PowerPoint offer to repair the file.
    """

    try:
        tc = cell._tc  # noqa: SLF001
        tc_pr = tc.get_or_add_tcPr()
        default = max(0.25, width_pt)
        outer = max(0.25, outer_width) if outer_width is not None else default
        inner = max(0.25, inner_width) if inner_width is not None else default
        end_row = min(row_count - 1, row + max(1, rowspan) - 1)
        end_col = min(col_count - 1, col + max(1, colspan) - 1)
        edge_widths = {
            "a:lnL": outer if col == 0 else inner,
            "a:lnR": outer if end_col >= col_count - 1 else inner,
            "a:lnT": outer if row == 0 else inner,
            "a:lnB": outer if end_row >= row_count - 1 else inner,
        }
        for edge, edge_width in edge_widths.items():
            for child in list(tc_pr):
                if child.tag == qn(edge):
                    tc_pr.remove(child)
            line = OxmlElement(edge)
            line.set("w", str(int(round(edge_width * EMU_PER_PT))))
            solid = OxmlElement("a:solidFill")
            srgb = OxmlElement("a:srgbClr")
            srgb.set("val", color_hex.lstrip("#").upper())
            solid.append(srgb)
            line.append(solid)
            _insert_tcpr_border(tc_pr, line)
    except Exception:  # noqa: BLE001
        logger.debug("Could not set cell borders", exc_info=True)


def _insert_tcpr_border(tc_pr: Any, line: Any) -> None:
    """Insert a border element before fills, keeping lnL→lnR→lnT→lnB order."""

    fill_tags = {qn(tag) for tag in _TCPR_FILL_TAGS}
    border_tags = [qn(tag) for tag in _TCPR_BORDER_TAGS]
    insert_at = len(tc_pr)
    for index, child in enumerate(tc_pr):
        if child.tag in fill_tags:
            insert_at = index
            break
    # Place among existing borders according to schema order.
    line_rank = border_tags.index(line.tag) if line.tag in border_tags else len(border_tags)
    for index, child in enumerate(list(tc_pr)[:insert_at]):
        if child.tag not in border_tags:
            continue
        child_rank = border_tags.index(child.tag)
        if child_rank > line_rank:
            insert_at = index
            break
    tc_pr.insert(insert_at, line)


def _clear_inherited_effects(shape: Any) -> None:
    """Neutralize theme shadows from python-pptx ``<p:style>`` effectRef.

    An empty ``<a:effectLst/>`` in spPr overrides the theme effect list, and
    ``effectRef idx="0"`` clears the style reference so LibreOffice / PowerPoint
    do not paint drop shadows on autoshapes or connectors.
    """

    try:
        element = shape._element  # noqa: SLF001
        sp_pr = getattr(element, "spPr", None)
        if sp_pr is not None:
            for child in list(sp_pr):
                if child.tag == qn("a:effectLst"):
                    sp_pr.remove(child)
            sp_pr.append(OxmlElement("a:effectLst"))
        style = element.find(qn("p:style"))
        if style is not None:
            for child in style:
                if child.tag == qn("a:effectRef"):
                    child.set("idx", "0")
    except Exception:  # noqa: BLE001
        logger.debug("Could not clear inherited shape effects", exc_info=True)


def _resolve_table_paragraph_align(align: str) -> PP_ALIGN:
    if align == "center":
        return PP_ALIGN.CENTER
    if align == "right":
        return PP_ALIGN.RIGHT
    return PP_ALIGN.LEFT


def _resolve_table_vertical_anchor(anchor: str) -> MSO_VERTICAL_ANCHOR:
    if anchor == "top":
        return MSO_VERTICAL_ANCHOR.TOP
    if anchor == "bottom":
        return MSO_VERTICAL_ANCHOR.BOTTOM
    return MSO_VERTICAL_ANCHOR.MIDDLE


def tcpr_children_in_schema_order(tc_pr: Any) -> bool:
    """Return True when border tags appear before fill tags (ECMA-376 order)."""

    seen_fill = False
    border_tags = {qn(tag) for tag in _TCPR_BORDER_TAGS}
    fill_tags = {qn(tag) for tag in _TCPR_FILL_TAGS}
    for child in tc_pr:
        if child.tag in fill_tags:
            seen_fill = True
        elif child.tag in border_tags and seen_fill:
            return False
    return True


def _normalize_tcpr_child_order(cell: Any) -> None:
    """Reorder tcPr children so borders precede fills."""

    try:
        tc = cell._tc  # noqa: SLF001
        tc_pr = tc.tcPr
        if tc_pr is None:
            return
        if tcpr_children_in_schema_order(tc_pr):
            return
        border_tags = {qn(tag) for tag in _TCPR_BORDER_TAGS}
        fill_tags = {qn(tag) for tag in _TCPR_FILL_TAGS}
        borders: list[Any] = []
        fills: list[Any] = []
        others: list[Any] = []
        for child in list(tc_pr):
            tc_pr.remove(child)
            if child.tag in border_tags:
                borders.append(child)
            elif child.tag in fill_tags:
                fills.append(child)
            else:
                others.append(child)
        border_order = {qn(tag): index for index, tag in enumerate(_TCPR_BORDER_TAGS)}
        borders.sort(key=lambda node: border_order.get(node.tag, 99))
        for child in borders + fills + others:
            tc_pr.append(child)
    except Exception:  # noqa: BLE001
        logger.debug("Could not normalize tcPr child order", exc_info=True)


def add_text_block(slide: Any, block: TextBlock, *, scale_x: float, scale_y: float) -> None:
    script = classify_text_script(block.text)
    resolved_font_family = block.font_family or default_font_family(script)
    resolved_typefaces = default_ppt_typefaces(script)
    if block.font_family:
        resolved_typefaces = {"latin": block.font_family, "ea": block.font_family, "cs": block.font_family}
    left, top, width, height = bbox_to_shape_geometry(block.bbox, scale_x, scale_y)
    textbox = slide.shapes.add_textbox(left, top, max(width, Emu(1)), max(height, Emu(1)))
    text_frame = textbox.text_frame
    text_frame.word_wrap = should_wrap_text_block(block)
    text_frame.vertical_anchor = resolve_vertical_anchor(block)
    text_frame.margin_left = 0
    text_frame.margin_right = 0
    text_frame.margin_top = 0
    text_frame.margin_bottom = 0
    text_frame.auto_size = MSO_AUTO_SIZE.NONE
    text_frame.clear()
    _set_text_frame_wrap(text_frame, should_wrap=should_wrap_text_block(block))

    paragraphs = list(getattr(block, "paragraphs", None) or [])
    if not paragraphs:
        paragraphs = [TextParagraph(text=block.text)]

    used_fit_text = False
    primary_run: Any | None = None
    for index, item in enumerate(paragraphs):
        paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.LEFT
        # Clear leftover runs on the first paragraph after text_frame.clear().
        if index == 0 and hasattr(paragraph, "clear"):
            paragraph.clear()
        run = paragraph.add_run()
        run.text = item.text
        if index == 0:
            primary_run = run
        font = run.font
        font.name = resolved_font_family
        _set_run_typefaces(run, resolved_typefaces)
        font_size = item.font_size if item.font_size is not None else block.font_size
        font_color = item.font_color if item.font_color is not None else block.font_color
        bold = block.bold if item.bold is None else item.bold
        italic = block.italic if item.italic is None else item.italic
        is_list_para = item.list_level is not None
        if font_size and (is_list_para or len(paragraphs) > 1):
            # List / multi-paragraph boxes never go through fit_text — set size now.
            font.size = Pt(max(6.0, font_size * min(scale_x, scale_y)))
        elif font_size and len(paragraphs) == 1:
            # Single plain paragraph still goes through OCR fit below.
            pass
        if font_color:
            font.color.rgb = RGBColor.from_string(font_color.lstrip("#"))
        font.bold = bold
        font.italic = italic
        if is_list_para:
            set_paragraph_bullet(
                paragraph,
                level=max(0, int(item.list_level or 0)),
                list_style=getattr(item, "list_style", None),
            )
        space_before = getattr(item, "space_before_pt", None)
        if space_before is not None and space_before > 0:
            _set_paragraph_space_before(paragraph, space_before * min(scale_x, scale_y))
        _set_paragraph_end_typefaces(paragraph, resolved_typefaces)

    if len(paragraphs) == 1 and paragraphs[0].list_level is None and primary_run is not None:
        used_fit_text = fit_text_frame(text_frame, block, scale_x=scale_x, scale_y=scale_y)
        _set_text_frame_wrap(text_frame, should_wrap=should_wrap_text_block(block))
        font = primary_run.font
        font.name = resolved_font_family
        _set_run_typefaces(primary_run, resolved_typefaces)
        if used_fit_text and block.source == "ocr" and "\n" not in block.text:
            safe_font_size_pt = resolve_fallback_font_size_pt(block, scale_x=scale_x, scale_y=scale_y)
            current_font_size_pt = getattr(font.size, "pt", None) if font.size is not None else None
            if current_font_size_pt is None or current_font_size_pt > safe_font_size_pt:
                font.size = Pt(safe_font_size_pt)
        if block.font_size and not used_fit_text:
            font.size = Pt(resolve_fallback_font_size_pt(block, scale_x=scale_x, scale_y=scale_y))
        if block.font_color:
            font.color.rgb = RGBColor.from_string(block.font_color.lstrip("#"))
        font.bold = block.bold
        font.italic = block.italic


def set_paragraph_bullet(
    paragraph: Any,
    *,
    level: int = 0,
    list_style: str | None = None,
) -> None:
    """Emit a real PowerPoint bullet or auto-number (not a literal glyph in run text)."""

    level = max(0, min(int(level), 8))
    paragraph.level = level
    try:
        p_pr = paragraph._p.get_or_add_pPr()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return
    for child in list(p_pr):
        tag = child.tag
        if tag in {qn("a:buFont"), qn("a:buChar"), qn("a:buNone"), qn("a:buAutoNum"), qn("a:buBlip")}:
            p_pr.remove(child)
    # Hanging indent so the bullet/number sits left of the text.
    indent_emu = 342900  # ~0.27"
    mar_l = indent_emu * (level + 1)
    p_pr.set("marL", str(mar_l))
    p_pr.set("indent", str(-indent_emu))

    style = (list_style or "bullet").strip()
    if style in {"arabicPeriod", "alphaLcPeriod", "alphaUcPeriod", "romanLcPeriod", "romanUcPeriod"}:
        bu_auto = OxmlElement("a:buAutoNum")
        bu_auto.set("type", style)
        p_pr.append(bu_auto)
        return

    bu_font = OxmlElement("a:buFont")
    bu_font.set("typeface", "Arial")
    bu_char = OxmlElement("a:buChar")
    bu_char.set("char", "•")
    p_pr.append(bu_font)
    p_pr.append(bu_char)


def _set_paragraph_space_before(paragraph: Any, space_before_pt: float) -> None:
    """Set a:spcBef from PDF line gaps so multi-paragraph boxes keep spacing."""

    try:
        p_pr = paragraph._p.get_or_add_pPr()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return
    for child in list(p_pr):
        if child.tag == qn("a:spcBef"):
            p_pr.remove(child)
    spc_bef = OxmlElement("a:spcBef")
    spc_pts = OxmlElement("a:spcPts")
    # OOXML spcPts is in hundredths of a point.
    spc_pts.set("val", str(int(round(max(0.0, space_before_pt) * 100))))
    spc_bef.append(spc_pts)
    p_pr.append(spc_bef)


def fit_text_frame(text_frame: Any, block: TextBlock, *, scale_x: float, scale_y: float) -> bool:
    if block.source != "ocr" or not block.text.strip():
        return False
    script = classify_text_script(block.text)
    font_path = choose_measurement_font(script, bold=block.bold)
    if font_path is None:
        logger.debug("No measurement font available for script=%s; skipping fit_text", script)
        return False

    font_family = block.font_family or default_font_family(script)
    base_size = max(6, int(round((block.font_size or 12.0) * min(scale_x, scale_y))))
    max_size = max(
        6,
        resolve_ocr_fit_max_size(
            block,
            font_path=font_path,
            base_size=base_size,
            scale_x=scale_x,
            scale_y=scale_y,
            script=script,
        ),
    )
    try:
        text_frame.fit_text(
            font_family=font_family,
            max_size=max_size,
            bold=block.bold,
            italic=block.italic,
            font_file=font_path,
        )
        return True
    except TypeError:
        logger.debug("python-pptx fit_text rejected OCR block %s", block.id)
        return False


def resolve_fallback_font_size_pt(block: TextBlock, *, scale_x: float, scale_y: float) -> float:
    base_size_pt = max(6.0, (block.font_size or 12.0) * min(scale_x, scale_y))
    if not block.text.strip():
        return base_size_pt

    script = classify_text_script(block.text)
    height_pt = max(1.0, (block.bbox[3] - block.bbox[1]) * scale_y)
    heuristic_cap = resolve_single_line_ocr_size_cap_pt(block.text, script, height_pt)
    font_path = choose_measurement_font(script, bold=block.bold)
    if font_path is None:
        return min(base_size_pt, heuristic_cap) if heuristic_cap is not None else base_size_pt

    width_pt = max(1.0, (block.bbox[2] - block.bbox[0]) * scale_x)
    width_limit = width_pt if "\n" in block.text else resolve_single_line_width_limit_pt(
        block.text,
        width_pt,
        script,
        height_pt=height_pt,
    )
    start_size = max(6, int(round(base_size_pt)))
    if is_title_like_single_line_ocr_block(block, width_pt=width_pt, height_pt=height_pt):
        start_size = max(
            start_size,
            resolve_ocr_fit_max_size(
                block,
                font_path=font_path,
                base_size=start_size,
                scale_x=scale_x,
                scale_y=scale_y,
                script=script,
            ),
        )
    for size in range(start_size, 5, -1):
        measured_width, measured_height = measure_text_dimensions(block.text, size, font_path)
        if measured_width <= width_limit and measured_height <= height_pt * 1.03:
            resolved_size = float(size)
            return min(resolved_size, heuristic_cap) if heuristic_cap is not None else resolved_size
    fallback_size = 6.0
    return min(fallback_size, heuristic_cap) if heuristic_cap is not None else fallback_size


def should_wrap_text_block(block: TextBlock) -> bool:
    if block.source == "ocr" and "\n" not in block.text:
        return False
    return True


def resolve_vertical_anchor(block: TextBlock) -> MSO_VERTICAL_ANCHOR:
    if block.source == "ocr" and "\n" not in block.text:
        return MSO_VERTICAL_ANCHOR.MIDDLE
    return MSO_VERTICAL_ANCHOR.TOP


def resolve_ocr_fit_max_size(
    block: TextBlock,
    *,
    font_path: str,
    base_size: int,
    scale_x: float,
    scale_y: float,
    script: str,
) -> int:
    height_pt = max(1.0, (block.bbox[3] - block.bbox[1]) * scale_y)
    max_size = min(96, max(base_size + 3, int(round(height_pt * ocr_fit_height_cap_ratio(script)))))
    heuristic_cap = resolve_single_line_ocr_size_cap_pt(block.text, script, height_pt)
    if heuristic_cap is not None:
        max_size = min(max_size, int(math.floor(heuristic_cap)))
    if "\n" in block.text:
        return max_size

    width_pt = max(1.0, (block.bbox[2] - block.bbox[0]) * scale_x)
    height_pt = max(1.0, (block.bbox[3] - block.bbox[1]) * scale_y)
    width_limit = resolve_single_line_width_limit_pt(block.text, width_pt, script, height_pt=height_pt)
    size_cap = max(6, max_size)
    for size in range(size_cap, 5, -1):
        measured_width, measured_height = measure_text_dimensions(block.text, size, font_path)
        if measured_width <= width_limit and measured_height <= height_pt * 1.03:
            return size
    return min(base_size, max_size)


def resolve_single_line_ocr_size_cap_pt(text: str, script: str, height_pt: float) -> float | None:
    if is_bracketed_ocr_label(text):
        return max(6.0, math.floor(height_pt * 0.66))
    return None


def resolve_single_line_width_limit_pt(text: str, width_pt: float, script: str, *, height_pt: float) -> float:
    ratio = single_line_fit_width_ratio(script)
    if is_bracketed_ocr_label(text):
        ratio = min(ratio, 0.87)
    if is_tiny_latin_footer_label(text, script, width_pt, height_pt):
        ratio = min(ratio, 0.90)
    return width_pt * ratio


def is_bracketed_ocr_label(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 4 or len(stripped) > 12:
        return False
    bracket_pairs = {
        "[": "]",
        "(": ")",
        "{": "}",
        "【": "】",
        "（": "）",
        "［": "］",
    }
    closing = bracket_pairs.get(stripped[0])
    return closing is not None and stripped.endswith(closing)


def is_tiny_latin_footer_label(text: str, script: str, width_pt: float, height_pt: float) -> bool:
    stripped = text.strip()
    if script != "latin":
        return False
    if len(stripped) < 6 or len(stripped) > 16:
        return False
    return width_pt <= 100.0 and height_pt <= 16.0


def is_title_like_single_line_ocr_block(block: TextBlock, *, width_pt: float, height_pt: float) -> bool:
    if block.source != "ocr" or "\n" in block.text:
        return False
    if block.block_role == "title":
        return True
    stripped = block.text.strip()
    if len(stripped) < 4 or len(stripped) > 36:
        return False
    if (block.font_size or 0.0) < 28.0:
        return False
    return width_pt >= 400.0 and height_pt >= 56.0


def bbox_to_shape_geometry(
    bbox: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
) -> tuple[Emu, Emu, Emu, Emu]:
    x0, y0, x1, y1 = bbox
    left = pt_to_emu(x0 * scale_x)
    top = pt_to_emu(y0 * scale_y)
    width = pt_to_emu(max(1.0, (x1 - x0) * scale_x))
    height = pt_to_emu(max(1.0, (y1 - y0) * scale_y))
    return left, top, width, height


def _set_text_frame_wrap(text_frame: Any, *, should_wrap: bool) -> None:
    if not hasattr(text_frame, "_txBody"):
        return
    body_properties = text_frame._txBody.bodyPr
    if should_wrap:
        body_properties.set("wrap", "square")
    else:
        body_properties.set("wrap", "none")


def _set_run_typefaces(run: Any, typefaces: dict[str, str]) -> None:
    if not hasattr(run, "_r"):
        return
    r_pr = run._r.get_or_add_rPr()
    for child_tag in (qn("a:latin"), qn("a:ea"), qn("a:cs")):
        for child in list(r_pr):
            if child.tag == child_tag:
                r_pr.remove(child)
    for tag_name, key in (("a:latin", "latin"), ("a:ea", "ea"), ("a:cs", "cs")):
        typeface = OxmlElement(tag_name)
        typeface.set("typeface", typefaces[key])
        r_pr.append(typeface)


def _set_paragraph_end_typefaces(paragraph: Any, typefaces: dict[str, str]) -> None:
    if not hasattr(paragraph, "_p"):
        return
    end_paragraph_properties = paragraph._p.get_or_add_endParaRPr()
    for child_tag in (qn("a:latin"), qn("a:ea"), qn("a:cs")):
        for child in list(end_paragraph_properties):
            if child.tag == child_tag:
                end_paragraph_properties.remove(child)
    for tag_name, key in (("a:latin", "latin"), ("a:ea", "ea"), ("a:cs", "cs")):
        typeface = OxmlElement(tag_name)
        typeface.set("typeface", typefaces[key])
        end_paragraph_properties.append(typeface)


def pt_to_emu(value: float) -> Emu:
    return Emu(int(round(value * EMU_PER_PT)))
