"""Tests for P1 digital-PDF vector and table nativization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz
from pptx import Presentation

from pdf2ppt.core import ConversionOptions
from pdf2ppt.models import TextBlock
from pdf2ppt.pipeline import convert_pdf
from pdf2ppt.ppt_render import render_page_to_slide
from pdf2ppt.render_compare import inspect_pptx_editability
from pdf2ppt.vector_nativization import (
    extract_tables,
    extract_vector_content,
    filter_text_blocks_outside_tables,
    remaining_visuals_after_nativization,
    should_attempt_vector_nativization,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _convert(fixture_name: str, *, vector_mode: str = "auto") -> tuple[Path, object]:
    work = Path(tempfile.mkdtemp(prefix="pdf2ppt-vector-"))
    pdf_path = FIXTURES_DIR / f"{fixture_name}.pdf"
    output_path = work / f"{fixture_name}.pptx"
    report_path = work / f"{fixture_name}.report.json"
    options = ConversionOptions(
        input_path=pdf_path,
        output_path=output_path,
        report_path=report_path,
        vector_mode=vector_mode,
        ocr_model_root=None,
    )
    report = convert_pdf(options)
    return output_path, report


class VectorModeGateTests(unittest.TestCase):
    def test_off_disables(self) -> None:
        ok, reason = should_attempt_vector_nativization(
            vector_mode="off",
            page_kind="digital",
            drawing_count=5,
            complexity_threshold=80,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "vector_mode=off")

    def test_scanned_disables(self) -> None:
        ok, reason = should_attempt_vector_nativization(
            vector_mode="auto",
            page_kind="scanned",
            drawing_count=5,
            complexity_threshold=80,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "scanned_page")

    def test_dense_page_exceeds_threshold(self) -> None:
        ok, reason = should_attempt_vector_nativization(
            vector_mode="auto",
            page_kind="digital",
            drawing_count=200,
            complexity_threshold=80,
        )
        self.assertFalse(ok)
        self.assertIsNotNone(reason)


class ShapeExtractionTests(unittest.TestCase):
    @unittest.skipUnless((FIXTURES_DIR / "geometric_chart.pdf").exists(), "fixture missing")
    def test_geometric_chart_extracts_bars_and_axes(self) -> None:
        with fitz.open(FIXTURES_DIR / "geometric_chart.pdf") as document:
            page = document[0]
            drawing_count = len(page.get_drawings())
            result = extract_vector_content(page, [], vector_mode="auto", page_kind="digital")
        kinds = [shape.kind for shape in result.shapes]
        self.assertIn("rect", kinds)
        self.assertIn("line", kinds)
        self.assertGreaterEqual(sum(1 for kind in kinds if kind == "rect"), 7)
        self.assertEqual(result.skipped_drawing_count, 0)
        self.assertFalse(
            remaining_visuals_after_nativization(
                image_boxes=[],
                drawing_count=drawing_count,
                nativized_drawing_count=len(result.nativized_drawing_indices),
                skipped_drawing_count=result.skipped_drawing_count,
            )
        )

    @unittest.skipUnless((FIXTURES_DIR / "vector_shapes.pdf").exists(), "fixture missing")
    def test_vector_shapes_fixture_covers_basic_kinds(self) -> None:
        with fitz.open(FIXTURES_DIR / "vector_shapes.pdf") as document:
            page = document[0]
            result = extract_vector_content(page, [], vector_mode="auto", page_kind="digital")
        kinds = {shape.kind for shape in result.shapes}
        self.assertIn("rect", kinds)
        self.assertIn("line", kinds)
        self.assertTrue({"ellipse", "round_rect", "arrow", "polygon", "freeform"} & kinds)


class TableExtractionTests(unittest.TestCase):
    @unittest.skipUnless((FIXTURES_DIR / "table.pdf").exists(), "fixture missing")
    def test_table_fixture_extracts_grid(self) -> None:
        with fitz.open(FIXTURES_DIR / "table.pdf") as document:
            page = document[0]
            tables = extract_tables(page, [], page_number=1)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table.row_count, 6)
        self.assertEqual(table.col_count, 4)
        texts = {cell.text for cell in table.cells}
        self.assertIn("Engineering", texts)
        self.assertIn("Department", texts)

    @unittest.skipUnless((FIXTURES_DIR / "merged_table.pdf").exists(), "fixture missing")
    def test_merged_table_detects_colspan(self) -> None:
        with fitz.open(FIXTURES_DIR / "merged_table.pdf") as document:
            page = document[0]
            tables = extract_tables(page, [], page_number=1)
        self.assertGreaterEqual(len(tables), 1)
        table = tables[0]
        merged = [cell for cell in table.cells if cell.colspan > 1 or cell.rowspan > 1]
        self.assertGreaterEqual(len(merged), 1)
        self.assertTrue(any("Americas" in cell.text or "Combined" in cell.text for cell in table.cells))


class PipelineIntegrationTests(unittest.TestCase):
    @unittest.skipUnless((FIXTURES_DIR / "geometric_chart.pdf").exists(), "fixture missing")
    def test_geometric_chart_native_shapes_no_raster_background(self) -> None:
        pptx_path, report = _convert("geometric_chart")
        page = report.pages[0]
        self.assertGreaterEqual(len(page.shape_elements), 8)
        self.assertEqual(page.background_mode, "elements")
        self.assertIsNone(page.background_image_bytes)
        metrics = inspect_pptx_editability(pptx_path)[0]
        self.assertGreaterEqual(metrics.auto_shape_count, 7)
        self.assertEqual(metrics.raster_background_area_ratio, 0.0)
        self.assertGreater(metrics.score, 0.5)

    @unittest.skipUnless((FIXTURES_DIR / "table.pdf").exists(), "fixture missing")
    def test_table_emits_native_table_without_duplicate_text(self) -> None:
        pptx_path, report = _convert("table")
        page = report.pages[0]
        self.assertEqual(len(page.table_elements), 1)
        # Title remains; cell texts absorbed into the table.
        self.assertTrue(all("Engineering" not in block.text for block in page.text_blocks))
        metrics = inspect_pptx_editability(pptx_path)[0]
        self.assertEqual(metrics.table_count, 1)
        self.assertEqual(metrics.raster_background_area_ratio, 0.0)

    @unittest.skipUnless((FIXTURES_DIR / "solid_background.pdf").exists(), "fixture missing")
    def test_vector_mode_off_preserves_raster_background(self) -> None:
        pptx_path, report = _convert("solid_background", vector_mode="off")
        page = report.pages[0]
        self.assertEqual(page.shape_elements, [])
        self.assertEqual(page.table_elements, [])
        self.assertIsNotNone(page.background_image_bytes)
        metrics = inspect_pptx_editability(pptx_path)[0]
        self.assertGreaterEqual(metrics.raster_background_area_ratio, 0.85)

    @unittest.skipUnless((FIXTURES_DIR / "flowchart.pdf").exists(), "fixture missing")
    def test_flowchart_fixture_nativizes_boxes_and_lines(self) -> None:
        pptx_path, report = _convert("flowchart")
        page = report.pages[0]
        self.assertGreaterEqual(len(page.shape_elements), 5)
        metrics = inspect_pptx_editability(pptx_path)[0]
        self.assertGreaterEqual(metrics.auto_shape_count + metrics.line_count, 5)
        self.assertEqual(metrics.raster_background_area_ratio, 0.0)


class RenderShapeTests(unittest.TestCase):
    def test_render_page_writes_shapes_and_tables(self) -> None:
        from pdf2ppt.models import PageResult, QualityScore, ShapeElement, TableCell, TableElement

        page_result = PageResult(
            page_number=1,
            page_kind="digital",
            background_mode="elements",
            width_pt=960,
            height_pt=540,
            text_blocks=[
                TextBlock(
                    id="t1",
                    source="native",
                    bbox=(40, 40, 200, 80),
                    text="Title",
                    confidence=1.0,
                )
            ],
            quality_score=QualityScore(1.0, 1.0, 1.0, 1.0),
            fallback_reason=None,
            shape_elements=[
                ShapeElement(
                    id="s1",
                    kind="rect",
                    bbox=(100, 100, 300, 200),
                    fill_color="#336699",
                    stroke_color="#000000",
                    stroke_width=1.0,
                ),
                ShapeElement(
                    id="s2",
                    kind="line",
                    bbox=(100, 250, 400, 250),
                    points=[(100, 250), (400, 250)],
                    stroke_color="#222222",
                    stroke_width=2.0,
                ),
            ],
            table_elements=[
                TableElement(
                    id="tbl1",
                    bbox=(100, 300, 500, 450),
                    row_count=2,
                    col_count=2,
                    cells=[
                        TableCell(0, 0, "A"),
                        TableCell(0, 1, "B"),
                        TableCell(1, 0, "C"),
                        TableCell(1, 1, "D"),
                    ],
                    column_widths=[200, 200],
                    row_heights=[75, 75],
                )
            ],
        )
        presentation = Presentation()
        presentation.slide_width = 12192000
        presentation.slide_height = 6858000
        render_page_to_slide(
            presentation,
            presentation.slide_layouts[6],
            page_result,
            slide_width_pt=960,
            slide_height_pt=540,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shapes.pptx"
            presentation.save(str(path))
            metrics = inspect_pptx_editability(path)[0]
        self.assertGreaterEqual(metrics.auto_shape_count, 1)
        self.assertGreaterEqual(metrics.line_count, 1)
        self.assertEqual(metrics.table_count, 1)
        self.assertEqual(metrics.text_box_count, 1)


class FilterTextTests(unittest.TestCase):
    def test_filter_absorbed_ids(self) -> None:
        blocks = [
            TextBlock(id="a", source="native", bbox=(0, 0, 1, 1), text="keep", confidence=1.0),
            TextBlock(id="b", source="native", bbox=(0, 0, 1, 1), text="drop", confidence=1.0),
        ]
        filtered = filter_text_blocks_outside_tables(blocks, {"b"})
        self.assertEqual([block.id for block in filtered], ["a"])


if __name__ == "__main__":
    unittest.main()
