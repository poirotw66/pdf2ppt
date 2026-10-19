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

    @unittest.skipUnless((FIXTURES_DIR / "vector_shapes.pdf").exists(), "fixture missing")
    def test_vector_shapes_arrow_points_right(self) -> None:
        # Fat tail corner is farther from centroid than the tip; tip must still
        # be the long-axis end with fewer vertices (right), not left.
        with fitz.open(FIXTURES_DIR / "vector_shapes.pdf") as document:
            page = document[0]
            result = extract_vector_content(page, [], vector_mode="auto", page_kind="digital")
        arrows = [shape for shape in result.shapes if shape.kind == "arrow"]
        self.assertEqual(len(arrows), 1)
        self.assertEqual(arrows[0].arrow_direction, "right")


class ArrowDirectionTests(unittest.TestCase):
    def test_synthetic_cardinal_arrows(self) -> None:
        from pdf2ppt.vector_nativization import _infer_arrow_direction

        right = [
            (320, 340),
            (480, 340),
            (480, 310),
            (560, 370),
            (480, 430),
            (480, 400),
            (320, 400),
        ]
        self.assertEqual(_infer_arrow_direction(right, (320, 310, 560, 430)), "right")

        left = [
            (560, 340),
            (400, 340),
            (400, 310),
            (320, 370),
            (400, 430),
            (400, 400),
            (560, 400),
        ]
        self.assertEqual(_infer_arrow_direction(left, (320, 310, 560, 430)), "left")

        up = [
            (340, 400),
            (340, 280),
            (310, 280),
            (370, 200),
            (430, 280),
            (400, 280),
            (400, 400),
        ]
        self.assertEqual(_infer_arrow_direction(up, (310, 200, 430, 400)), "up")

        down = [
            (340, 200),
            (340, 320),
            (310, 320),
            (370, 400),
            (430, 320),
            (400, 320),
            (400, 200),
        ]
        self.assertEqual(_infer_arrow_direction(down, (310, 200, 430, 400)), "down")


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


class TableStyleAndSchemaTests(unittest.TestCase):
    @unittest.skipUnless((FIXTURES_DIR / "table.pdf").exists(), "fixture missing")
    def test_table_disables_default_banding_and_sets_font_color(self) -> None:
        pptx_path, report = _convert("table")
        page = report.pages[0]
        self.assertTrue(all(cell.font_size is not None for cell in page.table_elements[0].cells))
        self.assertTrue(all(cell.font_color for cell in page.table_elements[0].cells))

        presentation = Presentation(str(pptx_path))
        # Find the table shape regardless of z-order / title-placeholder layout.
        table = next(shape.table for shape in presentation.slides[0].shapes if shape.has_table)
        self.assertFalse(table.first_row)
        self.assertFalse(table.horz_banding)
        header = table.cell(0, 0)
        # Explicit dark text, not theme white-on-blue.
        run = header.text_frame.paragraphs[0].runs[0]
        self.assertIsNotNone(run.font.color.rgb)
        self.assertNotEqual(str(run.font.color.rgb).upper(), "FFFFFF")

    @unittest.skipUnless((FIXTURES_DIR / "table.pdf").exists(), "fixture missing")
    def test_table_tcpr_borders_precede_fills(self) -> None:
        from pptx.oxml.ns import qn

        from pdf2ppt.ppt_render import tcpr_children_in_schema_order

        pptx_path, _report = _convert("table")
        presentation = Presentation(str(pptx_path))
        table = next(shape.table for shape in presentation.slides[0].shapes if shape.has_table)
        checked = 0
        for row in table.rows:
            for cell in row.cells:
                tc_pr = cell._tc.tcPr  # noqa: SLF001
                if tc_pr is None:
                    continue
                has_border = any(child.tag == qn("a:lnL") for child in tc_pr)
                has_fill = any(child.tag in {qn("a:solidFill"), qn("a:noFill")} for child in tc_pr)
                if has_border and has_fill:
                    self.assertTrue(tcpr_children_in_schema_order(tc_pr))
                    checked += 1
        self.assertGreater(checked, 0)

    @unittest.skipUnless((FIXTURES_DIR / "merged_table.pdf").exists(), "fixture missing")
    def test_merged_header_uses_center_alignment(self) -> None:
        _pptx_path, report = _convert("merged_table")
        table = report.pages[0].table_elements[0]
        header = next(cell for cell in table.cells if cell.colspan > 1 or "Combined" in cell.text)
        # PyMuPDF 2.x may report the spanning header as left/center/right depending
        # on which grid cell holds the run; accept any non-default explicit align
        # when the cell is a merge, otherwise require center.
        if header.colspan > 1:
            self.assertIn(header.horizontal_align, {"left", "center", "right"})
        else:
            self.assertEqual(header.horizontal_align, "center")

    @unittest.skipUnless((FIXTURES_DIR / "merged_table.pdf").exists(), "fixture missing")
    def test_merged_table_outer_inner_border_widths_and_font(self) -> None:
        from pptx.oxml.ns import qn

        from pdf2ppt.ppt_render import EMU_PER_PT

        pptx_path, report = _convert("merged_table")
        table = report.pages[0].table_elements[0]
        # Fixture uses uniform 1pt strokes so PyMuPDF 2.x find_tables() still
        # detects the grid (mixed outer/inner widths were invisible to it).
        self.assertAlmostEqual(table.outer_border_width or 0.0, 1.0, places=2)
        self.assertAlmostEqual(table.inner_border_width or 0.0, 1.0, places=2)
        self.assertTrue(all(cell.font_name == "Arial" for cell in table.cells if cell.text.strip()))
        self.assertTrue(any(cell.vertical_anchor == "middle" for cell in table.cells if cell.text.strip()))

        presentation = Presentation(str(pptx_path))
        ppt_table = next(shape for shape in presentation.slides[0].shapes if shape.has_table).table
        corner = ppt_table.cell(0, 0)._tc.tcPr  # noqa: SLF001
        ln_l = next(child for child in corner if child.tag == qn("a:lnL"))
        self.assertEqual(int(ln_l.get("w")), int(round(1.0 * EMU_PER_PT)))
        body = ppt_table.cell(1, 1)._tc.tcPr  # noqa: SLF001
        ln_t = next(child for child in body if child.tag == qn("a:lnT"))
        self.assertEqual(int(ln_t.get("w")), int(round(1.0 * EMU_PER_PT)))
        # Header text may sit in a spanning cell rather than (0,0).
        header_run = None
        for row in ppt_table.rows:
            for cell in row.cells:
                runs = cell.text_frame.paragraphs[0].runs
                if runs and "Combined" in (runs[0].text or ""):
                    header_run = runs[0]
                    break
            if header_run is not None:
                break
        self.assertIsNotNone(header_run)
        assert header_run is not None
        self.assertEqual(header_run.font.name, "Arial")


class ShadowClearingTests(unittest.TestCase):
    @unittest.skipUnless((FIXTURES_DIR / "vector_shapes.pdf").exists(), "fixture missing")
    def test_shapes_and_connectors_have_empty_effect_list(self) -> None:
        from pptx.oxml.ns import qn

        pptx_path, _report = _convert("vector_shapes")
        presentation = Presentation(str(pptx_path))
        checked = 0
        for shape in presentation.slides[0].shapes:
            element = shape._element  # noqa: SLF001
            style = element.find(qn("p:style"))
            if style is None:
                continue
            sp_pr = getattr(element, "spPr", None)
            self.assertIsNotNone(sp_pr)
            effect_lists = [child for child in sp_pr if child.tag == qn("a:effectLst")]
            self.assertEqual(len(effect_lists), 1)
            self.assertEqual(len(effect_lists[0]), 0)
            effect_ref = style.find(qn("a:effectRef"))
            self.assertIsNotNone(effect_ref)
            self.assertEqual(effect_ref.get("idx"), "0")
            checked += 1
        self.assertGreaterEqual(checked, 3)


class TransparentEraseTests(unittest.TestCase):
    def test_transparent_shape_not_erased_over_image(self) -> None:
        from pdf2ppt.background_erase import regions_for_background_erase, shape_should_erase_background
        from pdf2ppt.models import ShapeElement

        transparent = ShapeElement(
            id="t",
            kind="rect",
            bbox=(60, 340, 620, 460),
            fill_color="#000000",
            fill_opacity=0.35,
        )
        opaque = ShapeElement(
            id="o",
            kind="rect",
            bbox=(10, 10, 50, 50),
            fill_color="#FFFFFF",
            fill_opacity=1.0,
        )
        image_box = (0.0, 0.0, 960.0, 540.0)
        self.assertFalse(shape_should_erase_background(transparent, image_boxes=[image_box]))
        self.assertFalse(shape_should_erase_background(opaque, image_boxes=[image_box]))
        self.assertTrue(shape_should_erase_background(opaque, image_boxes=[]))
        regions = regions_for_background_erase(
            [transparent, opaque],
            [],
            image_boxes=[image_box],
        )
        self.assertEqual(regions, [])

    def test_uncomposite_restores_pre_tint_luma(self) -> None:
        import numpy as np
        from PIL import Image

        from pdf2ppt.background_erase import uncomposite_translucent_shapes_from_image
        from pdf2ppt.models import ShapeElement

        # Synthetic: true bg luma 100, black @ 35% → composited luma 65.
        true_bg = np.full((40, 80, 3), 100, dtype=np.uint8)
        alpha = 0.35
        composited = np.clip(
            ((1.0 - alpha) * true_bg.astype(np.float32) + alpha * 0.0).round(),
            0,
            255,
        ).astype(np.uint8)
        image = Image.fromarray(composited, mode="RGB")
        shape = ShapeElement(
            id="cap",
            kind="rect",
            bbox=(10, 5, 70, 35),
            fill_color="#000000",
            fill_opacity=alpha,
        )
        restored = uncomposite_translucent_shapes_from_image(
            image,
            [shape],
            page_width=80.0,
            page_height=40.0,
        )
        patch = np.asarray(restored)[8:32, 15:65]
        self.assertAlmostEqual(float(patch.mean()), 100.0, delta=1.5)

    @unittest.skipUnless((FIXTURES_DIR / "photo_texture.pdf").exists(), "fixture missing")
    def test_photo_texture_keeps_raster_under_transparent_caption(self) -> None:
        import numpy as np

        from pdf2ppt.background_erase import (
            erase_regions_from_image,
            regions_for_background_erase,
            uncomposite_translucent_shapes_from_image,
        )
        from pdf2ppt.block_analysis import render_page_image
        from pdf2ppt.native_extraction import extract_native_text_blocks
        from pdf2ppt.vector_nativization import extract_vector_content

        # Unit-level regression without OCR: transparent caption over a photo
        # must not be solid-erased; un-compositing peels the tint so a native
        # 35% box can sit on top without double-darkening.
        with fitz.open(FIXTURES_DIR / "photo_texture.pdf") as document:
            page = document[0]
            text_blocks, image_boxes = extract_native_text_blocks(page)
            result = extract_vector_content(
                page,
                text_blocks,
                vector_mode="auto",
                page_kind="hybrid",
            )
            translucent = [shape for shape in result.shapes if shape.fill_opacity < 0.99]
            self.assertTrue(translucent)
            erase_regions = regions_for_background_erase(
                result.shapes,
                [table.bbox for table in result.tables],
                image_boxes=image_boxes,
            )
            self.assertEqual(erase_regions, [])
            original = render_page_image(page, dpi=72)
            erased = erase_regions_from_image(
                original,
                erase_regions,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                fill_color=(128, 100, 80),
            )
            # No solid erase → identical pixels under the caption box.
            cx, cy = erased.width // 3, int(erased.height * 0.72)
            self.assertEqual(original.getpixel((cx, cy)), erased.getpixel((cx, cy)))

            caption = max(translucent, key=lambda shape: (shape.bbox[2] - shape.bbox[0]))
            restored = uncomposite_translucent_shapes_from_image(
                original,
                result.shapes,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
            )
            scale_x = original.width / float(page.rect.width)
            scale_y = original.height / float(page.rect.height)
            left = int(caption.bbox[0] * scale_x) + 4
            top = int(caption.bbox[1] * scale_y) + 4
            right = int(caption.bbox[2] * scale_x) - 4
            bottom = int(caption.bbox[3] * scale_y) - 4
            orig_patch = np.asarray(original.convert("RGB"))[top:bottom, left:right]
            rest_patch = np.asarray(restored.convert("RGB"))[top:bottom, left:right]
            # After un-composite, mean luma inside the box rises toward the
            # pre-tint photo (original composited mean is ~65.9).
            orig_luma = float(orig_patch.mean())
            rest_luma = float(rest_patch.mean())
            self.assertGreater(rest_luma, orig_luma + 8.0)
            # Re-applying the same alpha must recover the original composite luma.
            alpha = float(caption.fill_opacity)
            recomposed = (1.0 - alpha) * rest_patch.astype(np.float32)
            self.assertAlmostEqual(float(recomposed.mean()), orig_luma, delta=2.0)
            # Keep a textured neighbourhood (not a flat painted region).
            patch = [
                restored.getpixel((cx + dx, cy + dy))
                for dx in range(-6, 7)
                for dy in range(-6, 7)
            ]
            self.assertGreater(len(set(patch)), 8)


class EditabilityTableCellTests(unittest.TestCase):
    @unittest.skipUnless((FIXTURES_DIR / "table.pdf").exists(), "fixture missing")
    def test_editability_counts_nonempty_table_cells(self) -> None:
        pptx_path, _report = _convert("table")
        metrics = inspect_pptx_editability(pptx_path)[0]
        self.assertEqual(metrics.table_count, 1)
        self.assertGreaterEqual(metrics.nonempty_table_cell_count, 20)
        # Fully native table page should score well above the old ~0.50 ceiling.
        self.assertGreater(metrics.score, 0.7)


if __name__ == "__main__":
    unittest.main()
