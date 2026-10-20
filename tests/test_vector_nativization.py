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
        self.assertEqual(header.horizontal_align, "center")

    @unittest.skipUnless((FIXTURES_DIR / "merged_table.pdf").exists(), "fixture missing")
    def test_merged_table_outer_inner_border_widths_and_font(self) -> None:
        from pptx.oxml.ns import qn

        from pdf2ppt.ppt_render import EMU_PER_PT

        pptx_path, report = _convert("merged_table")
        table = report.pages[0].table_elements[0]
        self.assertAlmostEqual(table.outer_border_width or 0.0, 1.5, places=2)
        self.assertAlmostEqual(table.inner_border_width or 0.0, 1.0, places=2)
        self.assertTrue(all(cell.font_name == "Arial" for cell in table.cells if cell.text.strip()))
        self.assertTrue(any(cell.vertical_anchor == "middle" for cell in table.cells if cell.text.strip()))

        presentation = Presentation(str(pptx_path))
        ppt_table = next(shape for shape in presentation.slides[0].shapes if shape.has_table).table
        corner = ppt_table.cell(0, 0)._tc.tcPr  # noqa: SLF001
        ln_l = next(child for child in corner if child.tag == qn("a:lnL"))
        self.assertEqual(int(ln_l.get("w")), int(round(1.5 * EMU_PER_PT)))
        # Interior edge of a non-corner cell should use the thinner rule.
        body = ppt_table.cell(1, 1)._tc.tcPr  # noqa: SLF001
        ln_t = next(child for child in body if child.tag == qn("a:lnT"))
        self.assertEqual(int(ln_t.get("w")), int(round(1.0 * EMU_PER_PT)))
        run = ppt_table.cell(0, 0).text_frame.paragraphs[0].runs[0]
        self.assertEqual(run.font.name, "Arial")


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

    def test_page_background_and_large_fill_do_not_erase_small_picture(self) -> None:
        from pdf2ppt.background_erase import regions_for_background_erase, shape_should_erase_background
        from pdf2ppt.models import ShapeElement

        page_bg = ShapeElement(
            id="bg",
            kind="rect",
            bbox=(0, 0, 960, 540),
            fill_color="#E8F0FF",
            fill_opacity=1.0,
            is_page_background=True,
        )
        # ~93% colour card that is NOT flagged page-background.
        large_card = ShapeElement(
            id="card",
            kind="rect",
            bbox=(20, 20, 940, 510),
            fill_color="#224466",
            fill_opacity=1.0,
            is_page_background=False,
        )
        # ~10% of page picture.
        picture = (400.0, 180.0, 700.0, 360.0)
        self.assertFalse(shape_should_erase_background(page_bg, image_boxes=[picture]))
        self.assertFalse(shape_should_erase_background(large_card, image_boxes=[picture]))
        self.assertEqual(
            regions_for_background_erase([page_bg, large_card], [], image_boxes=[picture]),
            [],
        )
        # Same large card with no pictures may still be erased.
        self.assertTrue(shape_should_erase_background(large_card, image_boxes=[]))

    def test_large_fill_overlapping_picture_renders_under_background(self) -> None:
        from pdf2ppt.background_erase import (
            promote_shapes_under_background,
            shape_should_render_under_background,
        )
        from pdf2ppt.models import ShapeElement

        large_card = ShapeElement(
            id="card",
            kind="rect",
            bbox=(20, 20, 940, 510),
            fill_color="#224466",
            fill_opacity=1.0,
            is_page_background=False,
        )
        small_card = ShapeElement(
            id="badge",
            kind="rect",
            bbox=(40, 40, 120, 80),
            fill_color="#FF0000",
            fill_opacity=1.0,
            is_page_background=False,
        )
        translucent = ShapeElement(
            id="cap",
            kind="rect",
            bbox=(60, 340, 620, 460),
            fill_color="#000000",
            fill_opacity=0.35,
            is_page_background=False,
        )
        picture = (400.0, 180.0, 700.0, 360.0)
        self.assertTrue(shape_should_render_under_background(large_card, image_boxes=[picture]))
        self.assertFalse(shape_should_render_under_background(small_card, image_boxes=[picture]))
        self.assertFalse(shape_should_render_under_background(translucent, image_boxes=[picture]))
        self.assertFalse(shape_should_render_under_background(large_card, image_boxes=[]))
        promoted = promote_shapes_under_background(
            [large_card, small_card, translucent],
            image_boxes=[picture],
        )
        self.assertEqual(promoted, 1)
        self.assertTrue(large_card.is_page_background)
        self.assertFalse(small_card.is_page_background)
        self.assertFalse(translucent.is_page_background)

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


class PageBackgroundFillTests(unittest.TestCase):
    def test_is_page_background_fill_ignores_stroke(self) -> None:
        from pdf2ppt.models import ShapeElement
        from pdf2ppt.vector_nativization import is_page_background_fill, split_page_background_stroke

        page_area = 960.0 * 540.0
        full = (0.0, 0.0, 960.0, 540.0)
        self.assertTrue(
            is_page_background_fill(
                full,
                page_area=page_area,
                fill_color="#FFFFFF",
                stroke_color=None,
            )
        )
        # Stroked full-page fills are still backgrounds (stroke splits off).
        self.assertTrue(
            is_page_background_fill(
                full,
                page_area=page_area,
                fill_color="#FFFFFF",
                stroke_color="#000000",
                stroke_width=1.0,
            )
        )
        self.assertFalse(
            is_page_background_fill(
                (0.0, 0.0, 400.0, 300.0),
                page_area=page_area,
                fill_color="#FFFFFF",
            )
        )
        stroked = ShapeElement(
            id="bg",
            kind="rect",
            bbox=full,
            fill_color="#FFFFFF",
            stroke_color="#111111",
            stroke_width=2.0,
            is_page_background=True,
            z_order=0,
        )
        parts = split_page_background_stroke(stroked)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].is_page_background)
        self.assertIsNone(parts[0].stroke_color)
        self.assertFalse(parts[1].is_page_background)
        self.assertIsNone(parts[1].fill_color)
        self.assertEqual(parts[1].stroke_color, "#111111")

    @unittest.skipUnless((FIXTURES_DIR / "figure_caption.pdf").exists(), "fixture missing")
    def test_figure_caption_marks_white_page_fill_as_background(self) -> None:
        with fitz.open(FIXTURES_DIR / "figure_caption.pdf") as document:
            page = document[0]
            result = extract_vector_content(page, [], vector_mode="auto", page_kind="digital")
        self.assertEqual(len(result.shapes), 1)
        self.assertTrue(result.shapes[0].is_page_background)
        self.assertEqual(result.shapes[0].fill_color, "#FFFFFF")

    @unittest.skipUnless((FIXTURES_DIR / "figure_caption.pdf").exists(), "fixture missing")
    def test_figure_caption_page_fill_does_not_cover_raster(self) -> None:
        from pptx.oxml.ns import qn

        pptx_path, report = _convert("figure_caption")
        page = report.pages[0]
        self.assertEqual(page.background_mode, "overlay")
        self.assertIsNotNone(page.background_image_bytes)
        self.assertTrue(any(shape.is_page_background for shape in page.shape_elements))

        presentation = Presentation(str(pptx_path))
        slide = presentation.slides[0]
        children = list(slide.shapes._spTree)  # noqa: SLF001
        pic_idxs = [i for i, child in enumerate(children) if child.tag == qn("p:pic")]
        # Full-page white fill may exist, but only under the overlay picture.
        rect_idxs = []
        for index, child in enumerate(children):
            if child.tag != qn("p:sp"):
                continue
            sp_pr = child.find(qn("p:spPr"))
            xfrm = None if sp_pr is None else sp_pr.find(qn("a:xfrm"))
            ext = None if xfrm is None else xfrm.find(qn("a:ext"))
            if ext is None:
                continue
            cx = int(ext.get("cx") or 0)
            cy = int(ext.get("cy") or 0)
            if cx >= int(960 * 12700 * 0.95) and cy >= int(540 * 12700 * 0.95):
                rect_idxs.append(index)
        self.assertTrue(pic_idxs)
        self.assertTrue(rect_idxs)
        self.assertLess(min(rect_idxs), min(pic_idxs))

    def test_render_interleaves_shapes_and_pictures_by_z_order(self) -> None:
        import io

        from PIL import Image
        from pptx.oxml.ns import qn

        from pdf2ppt.models import ImagePlacement, PageResult, QualityScore, ShapeElement

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color=(10, 120, 200)).save(buffer, format="PNG")
        page_result = PageResult(
            page_number=1,
            page_kind="digital",
            background_mode="elements",
            width_pt=960,
            height_pt=540,
            text_blocks=[],
            quality_score=QualityScore(1.0, 1.0, 1.0, 1.0),
            fallback_reason=None,
            shape_elements=[
                ShapeElement(
                    id="early",
                    kind="rect",
                    bbox=(40, 40, 120, 100),
                    fill_color="#AABBCC",
                    z_order=1,
                ),
                ShapeElement(
                    id="late",
                    kind="rect",
                    bbox=(200, 200, 280, 260),
                    fill_color="#112233",
                    z_order=5,
                ),
            ],
            image_elements=[
                ImagePlacement(bbox=(140, 120, 300, 220), png_bytes=buffer.getvalue(), z_order=3),
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
        children = list(presentation.slides[0].shapes._spTree)  # noqa: SLF001
        order: list[str] = []
        for child in children:
            tag = child.tag
            if tag == qn("p:pic"):
                order.append("pic")
            elif tag == qn("p:sp"):
                order.append("sp")
        self.assertEqual(order, ["sp", "pic", "sp"])

    def test_bordered_page_fill_keeps_title_and_figure_visible(self) -> None:
        """Stroked full-page fill: fill under raster, outline above, title on top."""

        import io

        from PIL import Image
        from pptx.oxml.ns import qn

        from pdf2ppt.models import PageResult, QualityScore, ShapeElement, TextBlock
        from pdf2ppt.vector_nativization import split_page_background_stroke

        stroked = ShapeElement(
            id="bg",
            kind="rect",
            bbox=(0, 0, 960, 540),
            fill_color="#FFFFFF",
            stroke_color="#000000",
            stroke_width=3.0,
            is_page_background=True,
            z_order=0,
        )
        shapes = split_page_background_stroke(stroked)
        buffer = io.BytesIO()
        # Distinct non-white figure baked into the overlay raster.
        Image.new("RGB", (64, 40), color=(30, 140, 220)).save(buffer, format="PNG")
        page = PageResult(
            page_number=1,
            page_kind="digital",
            background_mode="overlay",
            width_pt=960,
            height_pt=540,
            background_image_bytes=buffer.getvalue(),
            shape_elements=shapes,
            text_blocks=[
                TextBlock(
                    id="title",
                    source="native",
                    bbox=(60, 30, 500, 80),
                    text="Bordered Architecture",
                    confidence=1.0,
                    font_size=26,
                    block_role="title",
                    use_title_placeholder=True,
                    reading_order=1,
                )
            ],
            quality_score=QualityScore(1, 1, 1, 1),
            fallback_reason=None,
        )
        presentation = Presentation()
        presentation.slide_width = 12192000
        presentation.slide_height = 6858000
        render_page_to_slide(
            presentation,
            presentation.slide_layouts[6],
            page,
            slide_width_pt=960,
            slide_height_pt=540,
        )
        slide = presentation.slides[0]
        children = list(slide.shapes._spTree)  # noqa: SLF001
        title_el = slide.shapes.title._element  # noqa: SLF001
        title_idx = children.index(title_el)
        pic_idxs = [i for i, child in enumerate(children) if child.tag == qn("p:pic")]
        sp_idxs = [i for i, child in enumerate(children) if child.tag == qn("p:sp") and child is not title_el]
        self.assertTrue(pic_idxs)
        self.assertGreaterEqual(len(sp_idxs), 2)
        # Page fill under picture; outline may sit above picture; title above all shapes.
        self.assertLess(min(sp_idxs), min(pic_idxs))
        self.assertGreater(title_idx, max(sp_idxs + pic_idxs))
        self.assertIn("Bordered Architecture", slide.shapes.title.text_frame.text)

    def test_small_picture_survives_full_page_fill_erase(self) -> None:
        """~10%-of-page picture on a full-page fill must not be painted away."""

        import numpy as np
        from PIL import Image

        from pdf2ppt.background_erase import (
            erase_regions_from_image,
            regions_for_background_erase,
            resolve_erase_fill_color,
        )
        from pdf2ppt.models import ShapeElement

        page_w, page_h = 960.0, 540.0
        # Synthetic page raster: white bg + blue picture patch (~10% area).
        raster = Image.new("RGB", (960, 540), color=(255, 255, 255))
        pix = raster.load()
        for y in range(180, 360):
            for x in range(400, 700):
                pix[x, y] = (20, 120, 200)
        page_bg = ShapeElement(
            id="bg",
            kind="rect",
            bbox=(0, 0, page_w, page_h),
            fill_color="#FFFFFF",
            fill_opacity=1.0,
            is_page_background=True,
        )
        picture_box = (400.0, 180.0, 700.0, 360.0)
        regions = regions_for_background_erase([page_bg], [], image_boxes=[picture_box])
        self.assertEqual(regions, [])
        erased = erase_regions_from_image(
            raster,
            regions,
            page_width=page_w,
            page_height=page_h,
            fill_color=resolve_erase_fill_color([page_bg], raster),
        )
        sample = np.asarray(erased)[200:300, 450:550]
        # Picture blue channel stays dominant — not painted white.
        self.assertGreater(float(sample[:, :, 2].mean()), 150.0)

    def test_card_below_page_background_cutoff_keeps_photo_visible(self) -> None:
        """~93% opaque card + ~10% photo: card under raster so photo stays visible.

        Generates the PDF in-process (no OCR / Paddle). Asserts paint order and
        that the orange photo region is not covered by the light-blue card fill.
        """

        import io

        import numpy as np
        from PIL import Image
        from pptx.oxml.ns import qn

        page_w, page_h = 960.0, 540.0
        # ~93% card (below PAGE_BACKGROUND_AREA_RATIO=0.95).
        card = fitz.Rect(17, 10, 943, 530)
        self.assertAlmostEqual(card.get_area() / (page_w * page_h), 0.93, delta=0.01)
        # Distinct orange photo (~10% of page).
        photo = fitz.Rect(560, 153, 860, 357)
        self.assertAlmostEqual(photo.get_area() / (page_w * page_h), 0.10, delta=0.02)

        texture = np.zeros((150, 220, 3), dtype=np.uint8)
        texture[:, :] = (230, 110, 40)
        noise = np.random.default_rng(0).integers(0, 25, size=(150, 220, 3), dtype=np.uint8)
        texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        png_buf = io.BytesIO()
        Image.fromarray(texture, mode="RGB").save(png_buf, format="PNG")

        doc = fitz.open()
        page = doc.new_page(width=page_w, height=page_h)
        page.draw_rect(card, color=None, fill=(0.85, 0.92, 1.0), fill_opacity=1)
        page.insert_image(photo, stream=png_buf.getvalue())
        page.insert_textbox(
            fitz.Rect(80, 60, 500, 110),
            "Card at 93% of the page",
            fontsize=28,
            fontname="helv",
            color=(0.1, 0.1, 0.15),
        )
        page.insert_textbox(
            fitz.Rect(80, 140, 520, 220),
            "This body paragraph sits on a large colored card.",
            fontsize=14,
            fontname="helv",
            color=(0.15, 0.15, 0.2),
        )
        pdf_bytes = doc.tobytes(garbage=4, deflate=True)
        doc.close()

        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-card93-"))
        pdf_path = work / "card_93pct.pdf"
        pptx_path = work / "card_93pct.pptx"
        pdf_path.write_bytes(pdf_bytes)
        options = ConversionOptions(
            input_path=pdf_path,
            output_path=pptx_path,
            report_path=work / "card_93pct.report.json",
            ocr_model_root=None,
            layout_engine="heuristic",
        )
        report = convert_pdf(options)
        page_result = report.pages[0]
        self.assertEqual(page_result.background_mode, "overlay")
        self.assertIsNotNone(page_result.background_image_bytes)
        self.assertTrue(page_result.shape_elements)
        card_shapes = [
            shape
            for shape in page_result.shape_elements
            if shape.fill_color and not (shape.stroke_color and shape.fill_color is None)
        ]
        self.assertTrue(any(shape.is_page_background for shape in card_shapes))

        presentation = Presentation(str(pptx_path))
        slide = presentation.slides[0]
        children = list(slide.shapes._spTree)  # noqa: SLF001
        pic_idxs = [i for i, child in enumerate(children) if child.tag == qn("p:pic")]
        card_idxs: list[int] = []
        for index, child in enumerate(children):
            if child.tag != qn("p:sp"):
                continue
            sp_pr = child.find(qn("p:spPr"))
            xfrm = None if sp_pr is None else sp_pr.find(qn("a:xfrm"))
            ext = None if xfrm is None else xfrm.find(qn("a:ext"))
            if ext is None:
                continue
            cx = int(ext.get("cx") or 0)
            cy = int(ext.get("cy") or 0)
            # Large card (~93% of 960×540 slide).
            if cx >= int(960 * 12700 * 0.90) and cy >= int(540 * 12700 * 0.90):
                card_idxs.append(index)
        self.assertTrue(pic_idxs, "expected full-slide background picture")
        self.assertTrue(card_idxs, "expected ~93% card auto-shape")
        # Card must sit under the raster that still contains the photo.
        self.assertLess(min(card_idxs), min(pic_idxs))

        # Title / body text stays above the background picture.
        text_idxs = []
        for index, child in enumerate(children):
            if child.tag != qn("p:sp"):
                continue
            if index in card_idxs:
                continue
            if child.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}t") is not None:
                text_idxs.append(index)
        self.assertTrue(text_idxs)
        self.assertGreater(min(text_idxs), min(pic_idxs))

    def test_dark_97pct_page_fill_stays_under_raster(self) -> None:
        """≥95% dark fill + photo: already page-background; keep under raster."""

        import io

        import numpy as np
        from PIL import Image
        from pptx.oxml.ns import qn

        page_w, page_h = 960.0, 540.0
        card = fitz.Rect(8, 6, 952, 534)
        self.assertGreaterEqual(card.get_area() / (page_w * page_h), 0.95)
        photo = fitz.Rect(600, 145, 880, 335)

        texture = np.full((120, 180, 3), (200, 90, 30), dtype=np.uint8)
        png_buf = io.BytesIO()
        Image.fromarray(texture, mode="RGB").save(png_buf, format="PNG")

        doc = fitz.open()
        page = doc.new_page(width=page_w, height=page_h)
        page.draw_rect(card, color=None, fill=(0.10, 0.12, 0.18), fill_opacity=1)
        page.insert_image(photo, stream=png_buf.getvalue())
        page.insert_textbox(
            fitz.Rect(60, 40, 500, 90),
            "Dark 97% backdrop",
            fontsize=28,
            fontname="helv",
            color=(0.95, 0.97, 1.0),
        )
        page.insert_textbox(
            fitz.Rect(60, 120, 560, 200),
            "Light body text on a dark backdrop that covers almost the whole page "
            "but not quite all of it, wrapping over multiple lines.",
            fontsize=14,
            fontname="helv",
            color=(0.90, 0.92, 0.96),
        )
        pdf_bytes = doc.tobytes(garbage=4, deflate=True)
        doc.close()

        work = Path(tempfile.mkdtemp(prefix="pdf2ppt-dark97-"))
        pdf_path = work / "dark_97pct.pdf"
        pptx_path = work / "dark_97pct.pptx"
        pdf_path.write_bytes(pdf_bytes)
        options = ConversionOptions(
            input_path=pdf_path,
            output_path=pptx_path,
            report_path=work / "dark_97pct.report.json",
            ocr_model_root=None,
            layout_engine="heuristic",
        )
        report = convert_pdf(options)
        page_result = report.pages[0]
        self.assertTrue(any(shape.is_page_background for shape in page_result.shape_elements))

        presentation = Presentation(str(pptx_path))
        children = list(presentation.slides[0].shapes._spTree)  # noqa: SLF001
        pic_idxs = [i for i, child in enumerate(children) if child.tag == qn("p:pic")]
        bg_idxs: list[int] = []
        for index, child in enumerate(children):
            if child.tag != qn("p:sp"):
                continue
            sp_pr = child.find(qn("p:spPr"))
            xfrm = None if sp_pr is None else sp_pr.find(qn("a:xfrm"))
            ext = None if xfrm is None else xfrm.find(qn("a:ext"))
            if ext is None:
                continue
            cx = int(ext.get("cx") or 0)
            cy = int(ext.get("cy") or 0)
            if cx >= int(960 * 12700 * 0.95) and cy >= int(540 * 12700 * 0.95):
                bg_idxs.append(index)
        self.assertTrue(pic_idxs and bg_idxs)
        self.assertLess(min(bg_idxs), min(pic_idxs))


class MultilineBodyWidthTests(unittest.TestCase):
    @unittest.skipUnless((FIXTURES_DIR / "large_mask.pdf").exists(), "fixture missing")
    def test_large_mask_text_box_stays_within_slide(self) -> None:
        """Assert box clamp using the PDF text layer only (no OCR / Paddle)."""

        from pdf2ppt.layout import LayoutPageResult, analyze_layout_heuristic, apply_layout_to_page
        from pdf2ppt.models import PageResult, QualityScore
        from pdf2ppt.native_extraction import extract_native_text_blocks
        from pdf2ppt.ppt_render import EMU_PER_PT, render_page_to_slide

        with fitz.open(FIXTURES_DIR / "large_mask.pdf") as document:
            page = document[0]
            native_blocks, _image_boxes = extract_native_text_blocks(page)
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)

        self.assertTrue(native_blocks)
        regions = analyze_layout_heuristic(native_blocks, page_width=page_width, page_height=page_height)
        applied = apply_layout_to_page(
            native_blocks,
            LayoutPageResult(regions=regions, engine="heuristic"),
            page_width=page_width,
            page_height=page_height,
        )
        page_result = PageResult(
            page_number=1,
            page_kind="digital",
            background_mode="elements",
            width_pt=page_width,
            height_pt=page_height,
            text_blocks=applied.text_blocks,
            quality_score=QualityScore(1.0, 1.0, 1.0, 1.0),
            fallback_reason=None,
        )
        presentation = Presentation()
        presentation.slide_width = int(round(page_width * EMU_PER_PT))
        presentation.slide_height = int(round(page_height * EMU_PER_PT))
        render_page_to_slide(
            presentation,
            presentation.slide_layouts[6],
            page_result,
            slide_width_pt=page_width,
            slide_height_pt=page_height,
        )
        slide = presentation.slides[0]
        slide_width_pt = float(presentation.slide_width) / EMU_PER_PT
        body = None
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            joined = "\n".join(paragraph.text for paragraph in shape.text_frame.paragraphs)
            if "quick brown fox" in joined:
                body = shape
                break
        self.assertIsNotNone(body)
        assert body is not None
        right_pt = float(body.left + body.width) / EMU_PER_PT
        self.assertLessEqual(right_pt, slide_width_pt + 0.5)
        # Wrap stays on so edits can reflow.
        self.assertTrue(body.text_frame.word_wrap)

    def test_lengthened_line_box_clamped_with_wrap_on(self) -> None:
        from pdf2ppt.models import TextBlock, TextParagraph
        from pdf2ppt.ppt_render import EMU_PER_PT, add_text_block

        presentation = Presentation()
        presentation.slide_width = int(960 * EMU_PER_PT)
        presentation.slide_height = int(540 * EMU_PER_PT)
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        base = "The quick brown fox jumps over the lazy dog while the committee reviews"
        long_line = base + (" extra clause" * 4)  # ~30%+ longer
        block = TextBlock(
            id="body",
            source="native",
            bbox=(60, 130, 860, 200),
            text=f"{long_line}\nSecond line stays shorter than the first one here",
            confidence=1.0,
            font_size=17,
            paragraphs=[
                TextParagraph(text=long_line, font_size=17, space_before_pt=None),
                TextParagraph(text="Second line stays shorter than the first one here", font_size=17, space_before_pt=4.0),
            ],
        )
        add_text_block(slide, block, scale_x=1.0, scale_y=1.0, slide_width_pt=960.0)
        textbox = [shape for shape in slide.shapes if getattr(shape, "has_text_frame", False)][-1]
        right_pt = float(textbox.left + textbox.width) / EMU_PER_PT
        self.assertLessEqual(right_pt, 960.0 - 11.0)  # slide width minus margin
        self.assertTrue(textbox.text_frame.word_wrap)


if __name__ == "__main__":
    unittest.main()
