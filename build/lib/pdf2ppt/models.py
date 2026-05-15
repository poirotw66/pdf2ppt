from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

BBox = tuple[float, float, float, float]
Polygon = tuple[tuple[float, float], ...]
PageKind = Literal["digital", "scanned", "hybrid"]
BackgroundMode = Literal["elements", "overlay", "full-page"]


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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class QualityScore:
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

    def to_report_dict(self) -> dict[str, object]:
        return {
            "page": self.page_number,
            "page_kind": self.page_kind,
            "background_mode": self.background_mode,
            "page_size_pt": [self.width_pt, self.height_pt],
            "quality_score": self.quality_score.to_dict(),
            "fallback_reason": self.fallback_reason,
            "background_inpaint_engine": self.background_inpaint_engine,
            "background_inpaint_note": self.background_inpaint_note,
            "text_blocks": [block.to_dict() for block in self.text_blocks],
        }


@dataclass(slots=True)
class ConversionReport:
    input_path: str
    output_path: str
    pages: list[PageResult]

    def to_dict(self) -> dict[str, object]:
        editable_pages = sum(
            1 for page in self.pages if page.background_mode in {"elements", "overlay"}
        )
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "page_count": len(self.pages),
            "editable_page_count": editable_pages,
            "pages": [page.to_report_dict() for page in self.pages],
        }
