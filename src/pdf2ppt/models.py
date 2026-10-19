from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

BBox = tuple[float, float, float, float]
Polygon = tuple[tuple[float, float], ...]
PageKind = Literal["digital", "scanned", "hybrid"]
BackgroundMode = Literal["elements", "overlay", "full-page"]
ShapeKind = Literal[
    "rect",
    "round_rect",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "freeform",
    "arrow",
]
ArrowDirection = Literal["left", "right", "up", "down"]
VectorMode = Literal["off", "auto", "aggressive"]
LayoutEngineName = Literal["heuristic", "ppstructure"]
# Normalized layout region labels used across heuristic + PP-StructureV3 paths.
LayoutLabel = Literal[
    "title",
    "paragraph_title",
    "text",
    "list",
    "figure",
    "table",
    "header",
    "footer",
    "page_number",
    "caption",
    "formula",
    "chart",
    "seal",
    "sidebar",
    "other",
]


@dataclass(slots=True)
class TextParagraph:
    """One paragraph inside a text box (supports real PowerPoint bullets)."""

    text: str
    list_level: int | None = None  # None = plain; 0+ = bullet indent level
    # bullet | arabicPeriod | alphaLcPeriod — drives buChar vs buAutoNum.
    list_style: str | None = None
    bold: bool | None = None
    italic: bool | None = None
    font_size: float | None = None
    font_color: str | None = None
    # Extra space before this paragraph (pt), from PDF line gaps.
    space_before_pt: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class LayoutRegion:
    """Typed page region from layout analysis (heuristic or PP-StructureV3)."""

    id: str
    label: LayoutLabel
    bbox: BBox
    confidence: float = 1.0
    reading_order: int = 0
    raw_label: str = ""
    engine: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TextBlock:
    id: str
    source: Literal["native", "ocr"]
    bbox: BBox
    text: str
    confidence: float
    font_family: str | None = None
    font_size: float | None = None
    font_color: str | None = None
    bold: bool = False
    italic: bool = False
    reading_order: int = 0
    block_role: str = "body"
    image_bbox: BBox | None = None
    image_polygon: Polygon | None = None
    # P2 layout fields -------------------------------------------------
    layout_region_id: str | None = None
    layout_label: str | None = None
    # When set, ppt_render emits these as separate paragraphs (bullets).
    paragraphs: list[TextParagraph] | None = None
    # Prefer mapping into the slide title placeholder when True.
    use_title_placeholder: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ShapeElement:
    """Native PowerPoint-ready vector shape extracted from a PDF drawing path."""

    id: str
    kind: ShapeKind
    bbox: BBox
    points: list[tuple[float, float]] = field(default_factory=list)
    fill_color: str | None = None
    stroke_color: str | None = None
    stroke_width: float | None = None
    stroke_dash: str | None = None
    fill_opacity: float = 1.0
    stroke_opacity: float = 1.0
    z_order: int = 0
    provenance: str = "native_drawing"
    confidence: float = 1.0
    locked: bool = True
    drawing_index: int | None = None
    is_page_background: bool = False
    arrow_direction: ArrowDirection | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TableCell:
    row: int
    col: int
    text: str
    rowspan: int = 1
    colspan: int = 1
    fill_color: str | None = None
    font_size: float | None = None
    font_color: str | None = None
    font_name: str | None = None
    bold: bool = False
    # "left" | "center" | "right"
    horizontal_align: str = "left"
    # "top" | "middle" | "bottom"
    vertical_anchor: str = "middle"
    text_block_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TableElement:
    """Native PowerPoint-ready table extracted via PyMuPDF ``find_tables()``."""

    id: str
    bbox: BBox
    row_count: int
    col_count: int
    cells: list[TableCell]
    column_widths: list[float]
    row_heights: list[float]
    border_color: str | None = None
    border_width: float | None = None
    # Optional distinct outer vs inner rule widths (PDF ruled tables).
    outer_border_width: float | None = None
    inner_border_width: float | None = None
    z_order: int = 0
    provenance: str = "native_table"
    confidence: float = 0.9
    locked: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class GroupElement:
    """Optional grouping of related native elements (SlideIR placeholder)."""

    id: str
    member_ids: list[str]
    bbox: BBox | None = None
    provenance: str = "native_group"
    confidence: float = 0.5
    locked: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class QualityScore:
    """Heuristic per-page scores produced during conversion.

    These are pipeline-time estimates (not render-and-compare). For slide-level
    fidelity / editability against a rendered PPTX, see ``render_compare`` and
    ``docs/slide-quality.md``.
    """

    text_confidence: float
    layout_overlap_score: float
    style_recovery_score: float
    editable_ratio: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ImagePlacement:
    bbox: BBox
    png_bytes: bytes = field(repr=False)


@dataclass(slots=True)
class PageResult:
    page_number: int
    page_kind: PageKind
    background_mode: BackgroundMode
    width_pt: float
    height_pt: float
    text_blocks: list[TextBlock]
    quality_score: QualityScore
    fallback_reason: str | None
    background_inpaint_engine: str | None = None
    background_inpaint_note: str | None = None
    background_image_bytes: bytes | None = field(default=None, repr=False)
    image_elements: list[ImagePlacement] = field(default_factory=list, repr=False)
    shape_elements: list[ShapeElement] = field(default_factory=list)
    table_elements: list[TableElement] = field(default_factory=list)
    group_elements: list[GroupElement] = field(default_factory=list)
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    vector_mode: str | None = None
    vector_fallback_reason: str | None = None
    layout_engine: str | None = None
    layout_fallback_reason: str | None = None

    def to_report_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "page": self.page_number,
            "page_kind": self.page_kind,
            "background_mode": self.background_mode,
            "page_size_pt": [self.width_pt, self.height_pt],
            "quality_score": self.quality_score.to_dict(),
            "fallback_reason": self.fallback_reason,
            "background_inpaint_engine": self.background_inpaint_engine,
            "background_inpaint_note": self.background_inpaint_note,
            "text_blocks": [block.to_dict() for block in self.text_blocks],
            "shape_elements": [shape.to_dict() for shape in self.shape_elements],
            "table_elements": [table.to_dict() for table in self.table_elements],
            "group_elements": [group.to_dict() for group in self.group_elements],
            "layout_regions": [region.to_dict() for region in self.layout_regions],
        }
        if self.vector_mode is not None:
            payload["vector_mode"] = self.vector_mode
        if self.vector_fallback_reason is not None:
            payload["vector_fallback_reason"] = self.vector_fallback_reason
        if self.layout_engine is not None:
            payload["layout_engine"] = self.layout_engine
        if self.layout_fallback_reason is not None:
            payload["layout_fallback_reason"] = self.layout_fallback_reason
        return payload


@dataclass(slots=True)
class ConversionReport:
    input_path: str
    output_path: str
    pages: list[PageResult]
    generative_inpaint: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        editable_pages = sum(
            1 for page in self.pages if page.background_mode in {"elements", "overlay"}
        )
        payload: dict[str, object] = {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "page_count": len(self.pages),
            "editable_page_count": editable_pages,
            "pages": [page.to_report_dict() for page in self.pages],
        }
        if self.generative_inpaint is not None:
            payload["generative_inpaint"] = self.generative_inpaint
        return payload
