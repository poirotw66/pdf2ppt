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
    _cell_bbox_from_geometry,
    apply_cell_fills,
    extract_tables,
    table_should_erase_background,
    transparent_table_text_mask_blocks,
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

    @unittest.skipUnless((FIXTURES_DIR / "two_column_bullets.pdf").exists(), "fixture missing")
    def test_two_column_bullets_fixture_is_not_a_table(self) -> None:
        """Negative fixture: Strengths/Risks bullet columns must not become a table."""

        with fitz.open(FIXTURES_DIR / "two_column_bullets.pdf") as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            self.assertEqual(extract_tables(page, blocks, page_number=1), [])

    def test_two_column_strengths_risks_bullets_are_not_a_table(self) -> None:
        """deck_eval13-style Strengths/Risks columns must stay text, not a 4×2 table."""

        def build(page: fitz.Page) -> None:
            page.insert_text(fitz.Point(60, 80), "Strengths and Risks", fontsize=22, fontname="helv")
            page.insert_text(fitz.Point(60, 120), "Strengths", fontsize=16, fontname="helv")
            page.insert_text(fitz.Point(500, 120), "Risks", fontsize=16, fontname="helv")
            for index, line in enumerate(
                [
                    "• Strong product-market fit across enterprise segments",
                    "• Diversified revenue from recurring subscriptions",
                    "• Experienced leadership with domain expertise",
                ]
            ):
                page.insert_text(fitz.Point(60, 160 + index * 36), line, fontsize=13, fontname="helv")
            for index, line in enumerate(
                [
                    "• Competitive pressure from larger platforms",
                    "• Dependency on a small number of key accounts",
                    "• Regulatory uncertainty in new markets",
                ]
            ):
                page.insert_text(fitz.Point(500, 160 + index * 36), line, fontsize=13, fontname="helv")

        path = _write_pdf(build)
        with fitz.open(path) as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            self.assertEqual(extract_tables(page, blocks, page_number=1), [])

    def test_borderless_column_edges_keep_left_aligned_text(self) -> None:
        """Column boundaries use text extents so left-aligned glyphs stay near source x."""

        col_x = [80.0, 300.0, 520.0]

        def build(page: fitz.Page) -> None:
            rows = [
                ["Name", "Dept", "Score"],
                ["Alice", "Eng", "92"],
                ["Bob", "Sales", "88"],
                ["Carol", "Ops", "95"],
                ["Dan", "HR", "91"],
            ]
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
        table = tables[0]
        self.assertEqual(table.col_count, 3)
        # Cell left edges should sit just left of each column's source x0.
        for col_index, source_x in enumerate(col_x):
            cell = next(cell for cell in table.cells if cell.col == col_index and cell.row == 1)
            geom = _cell_bbox_from_geometry(table, cell)
            self.assertLessEqual(geom[0], source_x)
            self.assertLess(source_x - geom[0], 12.0)
            self.assertEqual(cell.horizontal_align, "left")

    @unittest.skipUnless((FIXTURES_DIR / "borderless_table.pdf").exists(), "fixture missing")
    def test_borderless_fixture_emits_pptx_table(self) -> None:
        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-borderless-"))
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

    @unittest.skipUnless((FIXTURES_DIR / "borderless_table.pdf").exists(), "fixture missing")
    def test_borderless_last_column_keeps_header_unwrapped(self) -> None:
        """Last column must be wide enough that 'Score' does not wrap (regression C1)."""

        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-borderless-wrap-"))
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
        table = report.pages[0].table_elements[0]
        last_width = table.column_widths[-1]
        header = next(cell for cell in table.cells if cell.row == 0 and cell.col == table.col_count - 1)
        self.assertEqual(header.text, "Score")
        mar_l = float(header.margin_left or 0.0)
        mar_r = float(header.margin_right or 0.0)
        # Usable width must exceed the PDF Helvetica advance for "Score" (~36.6pt).
        self.assertGreaterEqual(last_width - mar_l - mar_r, 40.0)
        prs = Presentation(str(pptx_path))
        ppt_table = next(shape.table for shape in prs.slides[0].shapes if shape.has_table)
        score_cell = ppt_table.cell(0, table.col_count - 1)
        self.assertEqual(score_cell.text.strip(), "Score")
        self.assertEqual(len(score_cell.text_frame.paragraphs), 1)

    @unittest.skipUnless((FIXTURES_DIR / "zebra_borderless_table.pdf").exists(), "fixture missing")
    def test_zebra_column_text_stays_near_source(self) -> None:
        """Zebra snap must not proportionally scale inner columns (regression C2)."""

        # Fixture draws text at x0+10 for columns [90, 260, 180, 140] starting at 70.
        source_x = [80.0, 170.0, 430.0, 610.0]
        with fitz.open(FIXTURES_DIR / "zebra_borderless_table.pdf") as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            tables = extract_tables(page, blocks, page_number=1)
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table.col_count, 4)
        self.assertAlmostEqual(sum(table.column_widths), table.bbox[2] - table.bbox[0], places=1)
        for col_index, expected_x in enumerate(source_x):
            cell = next(cell for cell in table.cells if cell.col == col_index and cell.row == 1)
            geom = _cell_bbox_from_geometry(table, cell)
            text_x = geom[0] + float(cell.margin_left or 0.0)
            self.assertLess(abs(text_x - expected_x), 3.0, msg=f"col {col_index}")
        # Borderless: no stroked rules on the IR table.
        self.assertEqual(table.border_width, 0.0)
        self.assertIsNone(table.border_color)


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

    @unittest.skipUnless((FIXTURES_DIR / "table_over_photo.pdf").exists(), "fixture missing")
    def test_table_over_photo_inpaints_absorbed_glyphs(self) -> None:
        """Ghost text must not remain under native cells when the photo underlay is kept."""

        from pdf2ppt.block_analysis import render_page_image

        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-table-photo-mask-"))
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
        page_result = report.pages[0]
        self.assertIsNotNone(page_result.background_image_bytes)
        assert page_result.background_image_bytes is not None
        background = Image.open(io.BytesIO(page_result.background_image_bytes)).convert("RGB")
        bg_dpi = max(72, int(round(background.width * 72.0 / max(page_result.width_pt, 1.0))))

        with fitz.open(FIXTURES_DIR / "table_over_photo.pdf") as document:
            page = document[0]
            blocks, image_boxes = extract_native_text_blocks(page)
            tables = extract_tables(page, blocks, page_number=1)
            masks = transparent_table_text_mask_blocks(tables, page, image_boxes)
            self.assertGreater(len(masks), 0)
            source = render_page_image(page, dpi=bg_dpi).convert("RGB")

        # Under a masked word box the background should diverge from the source
        # glyphs (inpainted) while remaining photo-like (not solid white).
        sample = masks[0].bbox
        scale_x = background.width / page_result.width_pt
        scale_y = background.height / page_result.height_pt
        left = max(0, int(sample[0] * scale_x))
        top = max(0, int(sample[1] * scale_y))
        right = min(background.width, int(sample[2] * scale_x))
        bottom = min(background.height, int(sample[3] * scale_y))
        src_right = min(source.width, right)
        src_bottom = min(source.height, bottom)
        bg_crop = np.asarray(background.crop((left, top, right, bottom)), dtype=np.float32)
        src_crop = np.asarray(source.crop((left, top, src_right, src_bottom)), dtype=np.float32)
        # Align shapes if rounding differs by a pixel.
        height = min(bg_crop.shape[0], src_crop.shape[0])
        width = min(bg_crop.shape[1], src_crop.shape[1])
        bg_crop = bg_crop[:height, :width]
        src_crop = src_crop[:height, :width]
        self.assertGreater(float(np.mean(np.abs(bg_crop - src_crop))), 5.0)
        # Not wiped to near-white (photo must remain).
        self.assertLess(float(np.mean(bg_crop)), 240.0)


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

    @unittest.skipUnless((FIXTURES_DIR / "merged_table.pdf").exists(), "fixture missing")
    def test_merged_table_header_stays_vertically_centred(self) -> None:
        """Tall merged headers with balanced gaps must not flip to top (regression E)."""

        with fitz.open(FIXTURES_DIR / "merged_table.pdf") as document:
            page = document[0]
            blocks, _images = extract_native_text_blocks(page)
            tables = extract_tables(page, blocks, page_number=1)
        self.assertGreaterEqual(len(tables), 1)
        header = next(
            cell
            for cell in tables[0].cells
            if "Americas" in cell.text or "Combined" in cell.text or cell.colspan > 1
        )
        self.assertEqual(header.vertical_anchor, "middle")

    @unittest.skipUnless((FIXTURES_DIR / "table.pdf").exists(), "fixture missing")
    def test_ruled_table_keeps_measured_or_default_margins(self) -> None:
        """Do not zero Office cell margins globally (regression D)."""

        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-ruled-margin-"))
        pptx_path = work / "table.pptx"
        report = convert_pdf(
            ConversionOptions(
                input_path=FIXTURES_DIR / "table.pdf",
                output_path=pptx_path,
                report_path=work / "report.json",
                ocr_model_root=None,
                layout_engine="heuristic",
            )
        )
        cell = report.pages[0].table_elements[0].cells[0]
        # Measured inset from source or None (Office default) — never forced 0 without measure.
        if cell.margin_left is not None:
            self.assertGreater(cell.margin_left, 0.5)
        prs = Presentation(str(pptx_path))
        ppt_table = next(shape.table for shape in prs.slides[0].shapes if shape.has_table)
        ppt_cell = ppt_table.cell(0, 0)
        # python-pptx reports EMUs; Office default left margin is ~91440 (0.1").
        self.assertGreater(int(ppt_cell.margin_left), 0)

    def test_centred_cells_keep_default_horizontal_margins(self) -> None:
        """Centred cells must not get a capped marL that shifts text (regression F)."""

        from pdf2ppt.vector_nativization import _style_from_spans

        # Wide cell with short centred text → large left_gap; must not set marL.
        cell_bbox = (0.0, 0.0, 200.0, 30.0)
        span_index = [
            {
                "bbox": (80.0, 8.0, 120.0, 22.0),
                "size": 12.0,
                "font_color": "#000000",
                "font_name": "Helvetica",
                "bold": False,
                "text": "42",
            }
        ]
        style = _style_from_spans(cell_bbox, span_index)
        self.assertEqual(style["horizontal_align"], "center")
        self.assertIsNone(style["margin_left"])
        self.assertIsNone(style["margin_right"])


if __name__ == "__main__":
    unittest.main()
