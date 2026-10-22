"""Borderless / zebra tables, tables over photos, and tall form-cell anchoring."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from pptx import Presentation

from pdf2ppt.core import ConversionOptions
from pdf2ppt.native_extraction import extract_native_text_blocks
from pdf2ppt.pipeline import convert_pdf
from pdf2ppt.render_compare import inspect_pptx_editability
from pdf2ppt.vector_nativization import (
    apply_cell_fills,
    extract_tables,
    table_should_erase_background,
    _cell_bbox_from_geometry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _write_pdf(build) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=960, height=540)
    build(page)
    work = Path(tempfile.mkdtemp(prefix="pdf2ppt-table-recon-"))
    path = work / "case.pdf"
    path.write_bytes(doc.tobytes(garbage=4, deflate=True))
    doc.close()
    return path


class BorderlessAndZebraDetectionTests(unittest.TestCase):
    def test_borderless_aligned_text_becomes_native_table(self) -> None:
        def build(page: fitz.Page) -> None:
            rows = [
                ["Name", "Dept", "Score"],
                ["Alice", "Eng", "92"],
                ["Bob", "Sales", "88"],
                ["Carol", "Ops", "95"],
                ["Dan", "HR", "91"],
            ]
            col_x = [60.0, 260.0, 460.0]
            for row_index, row in enumerate(rows):
                for col_index, value in enumerate(row):
                    page.insert_text(
                        fitz.Point(col_x[col_index], 80 + row_index * 28 + 18),
                        value,
                        fontsize=14,
                        fontname="helv",
                    )

        path = _write_pdf(build)
        with fitz.open(path) as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            tables = extract_tables(page, blocks, page_number=1)
        self.assertEqual(len(tables), 1)
        self.assertGreaterEqual(tables[0].row_count, 4)
        self.assertEqual(tables[0].col_count, 3)
        self.assertEqual(tables[0].provenance, "inferred_alignment_table")
        texts = {cell.text for cell in tables[0].cells}
        self.assertIn("Alice", texts)
        self.assertIn("Score", texts)

    def test_zebra_borderless_keeps_stripe_fills(self) -> None:
        def build(page: fitz.Page) -> None:
            rows = [["ID", "Name", "Status"]] + [
                [str(index), f"Item{index}", "OK"] for index in range(1, 6)
            ]
            col_widths = [80.0, 200.0, 120.0]
            x0, y0, row_h = 60.0, 70.0, 30.0
            for row_index, row in enumerate(rows):
                fill = (0.2, 0.2, 0.25) if row_index == 0 else (
                    (0.9, 0.94, 1.0) if row_index % 2 else (1.0, 1.0, 1.0)
                )
                page.draw_rect(
                    fitz.Rect(x0, y0 + row_index * row_h, x0 + sum(col_widths), y0 + (row_index + 1) * row_h),
                    color=None,
                    fill=fill,
                )
                x = x0
                for col_index, value in enumerate(row):
                    page.insert_text(
                        fitz.Point(x + 8, y0 + row_index * row_h + 20),
                        value,
                        fontsize=13,
                        fontname="helv",
                        color=(1, 1, 1) if row_index == 0 else (0, 0, 0),
                    )
                    x += col_widths[col_index]

        path = _write_pdf(build)
        with fitz.open(path) as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            tables = extract_tables(page, blocks, page_number=1)
            self.assertEqual(len(tables), 1)
            apply_cell_fills(tables[0], page)
            header = {cell.fill_color for cell in tables[0].cells if cell.row == 0}
            odd = {cell.fill_color for cell in tables[0].cells if cell.row > 0 and cell.row % 2 == 1}
            even = {cell.fill_color for cell in tables[0].cells if cell.row > 0 and cell.row % 2 == 0}
        self.assertEqual(header, {"#333340"})
        self.assertEqual(odd, {"#E5F0FF"})
        self.assertEqual(even, {"#FFFFFF"})

    def test_two_column_body_copy_is_not_a_table(self) -> None:
        def build(page: fitz.Page) -> None:
            left = ("The quick brown fox jumps over the lazy dog while stakeholders review. " * 4)
            right = ("Native rendering keeps text boxes editable on the right column today. " * 4)
            page.insert_textbox(fitz.Rect(40, 80, 450, 480), left, fontsize=14, fontname="helv")
            page.insert_textbox(fitz.Rect(500, 80, 920, 480), right, fontsize=14, fontname="helv")

        path = _write_pdf(build)
        with fitz.open(path) as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            self.assertEqual(extract_tables(page, blocks, page_number=1), [])

    def test_bullet_list_is_not_a_table(self) -> None:
        def build(page: fitz.Page) -> None:
            for index, label in enumerate(
                ["Kickoff and goals", "Market update", "Roadmap review", "Q&A session", "Next steps"]
            ):
                page.insert_text(
                    fitz.Point(60, 100 + index * 36),
                    f"• {label}",
                    fontsize=16,
                    fontname="helv",
                )

        path = _write_pdf(build)
        with fitz.open(path) as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            self.assertEqual(extract_tables(page, blocks, page_number=1), [])

    def test_card_layout_is_not_a_table(self) -> None:
        def build(page: fitz.Page) -> None:
            for index, title in enumerate(["Alpha", "Beta", "Gamma"]):
                x = 60 + index * 300
                page.draw_rect(
                    fitz.Rect(x, 120, x + 260, 360),
                    color=(0.7, 0.7, 0.7),
                    fill=(0.95, 0.95, 0.98),
                    width=1,
                )
                page.insert_text(fitz.Point(x + 20, 160), title, fontsize=18, fontname="helv")
                page.insert_textbox(
                    fitz.Rect(x + 20, 180, x + 240, 330),
                    "Card body copy describing a feature highlight for demos.",
                    fontsize=12,
                    fontname="helv",
                )

        path = _write_pdf(build)
        with fitz.open(path) as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            self.assertEqual(extract_tables(page, blocks, page_number=1), [])

    @unittest.skipUnless((FIXTURES_DIR / "borderless_table.pdf").exists(), "fixture missing")
    def test_borderless_fixture_emits_pptx_table(self) -> None:
        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-borderless-fix"))
        pptx_path = work / "borderless.pptx"
        report = convert_pdf(
            ConversionOptions(
                input_path=FIXTURES_DIR / "borderless_table.pdf",
                output_path=pptx_path,
                report_path=work / "report.json",
                ocr_model_root=None,
                layout_engine="heuristic",
            )
        )
        self.assertTrue(report.pages[0].table_elements)
        metrics = inspect_pptx_editability(pptx_path)[0]
        self.assertGreaterEqual(metrics.table_count, 1)


class TableOverPhotoTests(unittest.TestCase):
    def test_cells_stay_transparent_over_photo(self) -> None:
        def build(page: fitz.Page) -> None:
            texture = np.zeros((540, 960, 3), dtype=np.uint8)
            texture[:, :, 0] = 40
            texture[:, :, 1] = 120
            texture[:, :, 2] = 200
            texture[::20] += 30
            buffer = io.BytesIO()
            Image.fromarray(texture, mode="RGB").save(buffer, format="PNG")
            page.insert_image(fitz.Rect(0, 0, 960, 540), stream=buffer.getvalue())
            x0, y0, row_h = 100.0, 100.0, 36.0
            col_widths = [150.0, 150.0, 150.0]
            rows = [["A", "B", "C"], ["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
            page.draw_rect(
                fitz.Rect(x0, y0, x0 + sum(col_widths), y0 + len(rows) * row_h),
                color=(1, 1, 1),
                width=1.5,
            )
            for index in range(1, len(rows)):
                page.draw_line(
                    fitz.Point(x0, y0 + index * row_h),
                    fitz.Point(x0 + sum(col_widths), y0 + index * row_h),
                    color=(1, 1, 1),
                    width=1,
                )
            for index in range(1, 3):
                page.draw_line(
                    fitz.Point(x0 + index * 150, y0),
                    fitz.Point(x0 + index * 150, y0 + len(rows) * row_h),
                    color=(1, 1, 1),
                    width=1,
                )
            for row_index, row in enumerate(rows):
                x = x0
                for value in row:
                    page.insert_text(
                        fitz.Point(x + 10, y0 + row_index * row_h + 24),
                        value,
                        fontsize=14,
                        fontname="helv",
                        color=(1, 1, 1),
                    )
                    x += 150.0

        path = _write_pdf(build)
        with fitz.open(path) as document:
            page = document[0]
            blocks, image_boxes = extract_native_text_blocks(page)
            tables = extract_tables(page, blocks, page_number=1)
            self.assertEqual(len(tables), 1)
            apply_cell_fills(tables[0], page, image_boxes=image_boxes)
            self.assertTrue(all(cell.fill_color is None for cell in tables[0].cells))
            self.assertFalse(table_should_erase_background(tables[0], image_boxes))

    @unittest.skipUnless((FIXTURES_DIR / "table_over_photo.pdf").exists(), "fixture missing")
    def test_table_over_photo_fixture_preserves_raster_under_table(self) -> None:
        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-table-photo-"))
        pptx_path = work / "table_over_photo.pptx"
        report = convert_pdf(
            ConversionOptions(
                input_path=FIXTURES_DIR / "table_over_photo.pdf",
                output_path=pptx_path,
                report_path=work / "report.json",
                ocr_model_root=None,
                layout_engine="heuristic",
            )
        )
        page = report.pages[0]
        self.assertTrue(page.table_elements)
        table = page.table_elements[0]
        self.assertTrue(all(cell.fill_color is None for cell in table.cells))
        self.assertIsNotNone(page.background_image_bytes)
        metrics = inspect_pptx_editability(pptx_path)[0]
        self.assertGreaterEqual(metrics.table_count, 1)
        # Background picture must remain (photo visible under transparent cells).
        prs = Presentation(str(pptx_path))
        slide = prs.slides[0]
        self.assertTrue(any(shape.shape_type is not None and shape.has_table for shape in slide.shapes))


class FormTallEmptyCellTests(unittest.TestCase):
    def test_tall_cell_text_uses_top_anchor(self) -> None:
        def build(page: fitz.Page) -> None:
            page.draw_rect(fitz.Rect(50, 50, 900, 500), color=(0, 0, 0), width=1)
            page.draw_line(fitz.Point(50, 100), fitz.Point(900, 100), color=(0, 0, 0))
            page.draw_line(fitz.Point(250, 50), fitz.Point(250, 500), color=(0, 0, 0))
            page.insert_text(fitz.Point(60, 80), "Label", fontsize=14, fontname="helv")
            page.insert_text(fitz.Point(260, 80), "Value", fontsize=14, fontname="helv")
            page.insert_text(
                fitz.Point(260, 120),
                "Comment near top of tall cell",
                fontsize=14,
                fontname="helv",
            )

        path = _write_pdf(build)
        with fitz.open(path) as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            tables = extract_tables(page, blocks, page_number=1)
        self.assertEqual(len(tables), 1)
        tall = next(cell for cell in tables[0].cells if "Comment" in cell.text)
        self.assertEqual(tall.vertical_anchor, "top")
        geom = _cell_bbox_from_geometry(tables[0], tall)
        self.assertGreater(geom[3] - geom[1], 200)

    @unittest.skipUnless((FIXTURES_DIR / "form_tall_empty_cell.pdf").exists(), "fixture missing")
    def test_form_fixture_keeps_top_anchor_in_pptx(self) -> None:
        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-form-tall-"))
        pptx_path = work / "form.pptx"
        report = convert_pdf(
            ConversionOptions(
                input_path=FIXTURES_DIR / "form_tall_empty_cell.pdf",
                output_path=pptx_path,
                report_path=work / "report.json",
                ocr_model_root=None,
                layout_engine="heuristic",
            )
        )
        table = report.pages[0].table_elements[0]
        comment = next(cell for cell in table.cells if "Comment" in cell.text)
        self.assertEqual(comment.vertical_anchor, "top")
        prs = Presentation(str(pptx_path))
        ppt_table = next(shape.table for shape in prs.slides[0].shapes if shape.has_table)
        found = False
        for row in ppt_table.rows:
            for cell in row.cells:
                if "Comment" in cell.text:
                    self.assertEqual(int(cell.vertical_anchor), 1)  # TOP
                    found = True
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
