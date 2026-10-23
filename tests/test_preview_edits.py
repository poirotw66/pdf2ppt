"""Unit tests for preview-before-convert edit overlay + payload builders."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pdf2ppt.ir_io import SlideIrError, default_assets_dir_name, write_slide_ir
from pdf2ppt.models import (
    ImagePlacement,
    PageResult,
    QualityScore,
    ShapeElement,
    TableCell,
    TableElement,
    TextBlock,
)
from pdf2ppt.preview_edits import (
    apply_preview_edits_to_slideir,
    build_preview_payload,
    materialize_raster_fallback_backgrounds,
    normalize_preview_edits,
    save_preview_edits,
)


def _tiny_png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _sample_page() -> PageResult:
    return PageResult(
        page_number=1,
        page_kind="digital",
        background_mode="elements",
        width_pt=320.0,
        height_pt=240.0,
        text_blocks=[
            TextBlock(
                id="t1",
                source="native",
                bbox=(40.0, 40.0, 200.0, 80.0),
                text="Hello",
                confidence=1.0,
                font_size=18.0,
                reading_order=0,
                block_role="body",
            ),
            TextBlock(
                id="t2",
                source="native",
                bbox=(40.0, 100.0, 200.0, 140.0),
                text="Body copy",
                confidence=1.0,
                font_size=12.0,
                reading_order=1,
                block_role="body",
            ),
        ],
        quality_score=QualityScore(1.0, 1.0, 1.0, 1.0),
        fallback_reason=None,
        image_elements=[ImagePlacement(bbox=(10.0, 160.0, 60.0, 200.0), png_bytes=_tiny_png())],
        shape_elements=[
            ShapeElement(id="s1", kind="rect", bbox=(20.0, 20.0, 80.0, 60.0), fill_color="#336699")
        ],
        table_elements=[
            TableElement(
                id="tbl1",
                bbox=(100.0, 150.0, 280.0, 220.0),
                row_count=2,
                col_count=2,
                cells=[
                    TableCell(row=0, col=0, text="A"),
                    TableCell(row=0, col=1, text="B"),
                    TableCell(row=1, col=0, text="C"),
                    TableCell(row=1, col=1, text="D"),
                ],
                column_widths=[90.0, 90.0],
                row_heights=[35.0, 35.0],
            )
        ],
    )


class PreviewEditsTests(unittest.TestCase):
    def test_normalize_rejects_invalid_role(self) -> None:
        with self.assertRaises(SlideIrError):
            normalize_preview_edits(
                {
                    "pages": [
                        {
                            "page": 1,
                            "role_overrides": {"t1": "subtitle"},
                        }
                    ]
                }
            )

    def test_normalize_rejects_duplicate_pages(self) -> None:
        with self.assertRaises(SlideIrError):
            normalize_preview_edits({"pages": [{"page": 1}, {"page": 1}]})

    def test_build_preview_payload_applies_edits_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ir_path = Path(tmp) / "deck.slideir.json"
            write_slide_ir([_sample_page()], ir_path, slide_width_pt=320.0, slide_height_pt=240.0)
            slideir = json.loads(ir_path.read_text(encoding="utf-8"))
            # Image ids are assigned during serialize.
            image_id = slideir["pages"][0]["images"][0]["id"]

            edits = {
                "pages": [
                    {
                        "page": 1,
                        "excluded_ids": ["t2", image_id],
                        "role_overrides": {"t1": "title"},
                        "raster_fallback": False,
                    }
                ]
            }
            payload = build_preview_payload(slideir, job_id="abc", edits=edits)
            page = payload["pages"][0]
            self.assertEqual(page["image_url"], "/jobs/abc/pages/1.jpg")
            by_id = {block["id"]: block for block in page["text_blocks"]}
            # IR-native role is preserved; UI applies role_overrides from payload["edits"].
            self.assertEqual(by_id["t1"]["block_role"], "body")
            self.assertEqual(payload["edits"]["pages"][0]["role_overrides"]["t1"], "title")
            self.assertFalse(by_id["t1"]["excluded"])
            self.assertTrue(by_id["t2"]["excluded"])
            image = next(item for item in page["images"] if item["id"] == image_id)
            self.assertTrue(image["excluded"])
            self.assertEqual(len(page["tables"][0]["cells"]), 4)
            self.assertEqual(len(page["shapes"]), 1)

    def test_apply_edits_excludes_and_changes_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ir_path = Path(tmp) / "deck.slideir.json"
            write_slide_ir([_sample_page()], ir_path, slide_width_pt=320.0, slide_height_pt=240.0)
            slideir = json.loads(ir_path.read_text(encoding="utf-8"))
            image_id = slideir["pages"][0]["images"][0]["id"]
            edits = {
                "pages": [
                    {
                        "page": 1,
                        "excluded_ids": ["t2", "s1", "tbl1", image_id],
                        "role_overrides": {"t1": "header"},
                        "raster_fallback": False,
                    }
                ]
            }
            edited = apply_preview_edits_to_slideir(slideir, edits)
            page = edited["pages"][0]
            self.assertEqual([b["id"] for b in page["text_blocks"]], ["t1"])
            self.assertEqual(page["text_blocks"][0]["block_role"], "header")
            self.assertEqual(page["shapes"], [])
            self.assertEqual(page["tables"], [])
            self.assertEqual(page["images"], [])

    def test_raster_fallback_clears_elements_and_sets_full_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ir_path = Path(tmp) / "deck.slideir.json"
            write_slide_ir([_sample_page()], ir_path, slide_width_pt=320.0, slide_height_pt=240.0)
            slideir = json.loads(ir_path.read_text(encoding="utf-8"))
            edits = {"pages": [{"page": 1, "excluded_ids": [], "role_overrides": {}, "raster_fallback": True}]}
            edited = apply_preview_edits_to_slideir(
                slideir,
                edits,
                ensure_full_page_background={1: b"jpeg-bytes"},
            )
            page = edited["pages"][0]
            self.assertEqual(page["background_mode"], "full-page")
            self.assertEqual(page["text_blocks"], [])
            self.assertEqual(page["shapes"], [])
            self.assertIsNotNone(page.get("background"))

            assets_root = Path(tmp) / default_assets_dir_name(ir_path)
            materialize_raster_fallback_backgrounds(
                edited,
                assets_root=assets_root,
                page_images={1: b"jpeg-bytes"},
            )
            bg_path = assets_root / "pages" / "001" / "background.jpg"
            self.assertTrue(bg_path.exists())
            self.assertEqual(bg_path.read_bytes(), b"jpeg-bytes")

    def test_save_and_reload_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview_edits.json"
            saved = save_preview_edits(
                path,
                {
                    "pages": [
                        {
                            "page": 2,
                            "excluded_ids": ["t1"],
                            "role_overrides": {"t9": "footer"},
                            "raster_fallback": True,
                        }
                    ]
                },
            )
            self.assertEqual(saved["pages"][0]["page"], 2)
            self.assertTrue(path.exists())

    def test_normalize_dedupes_excluded_ids(self) -> None:
        normalized = normalize_preview_edits(
            {"pages": [{"page": 1, "excluded_ids": ["a", "b", "a", "b"], "role_overrides": {}}]}
        )
        self.assertEqual(normalized["pages"][0]["excluded_ids"], ["a", "b"])

    def test_normalize_rejects_too_many_excluded_ids(self) -> None:
        with self.assertRaises(SlideIrError):
            normalize_preview_edits(
                {"pages": [{"page": 1, "excluded_ids": [f"id_{i}" for i in range(20_001)]}]}
            )

    def test_overlay_page_sets_baked_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = _sample_page()
            page.background_mode = "overlay"
            ir_path = Path(tmp) / "deck.slideir.json"
            write_slide_ir([page], ir_path, slide_width_pt=320.0, slide_height_pt=240.0)
            slideir = json.loads(ir_path.read_text(encoding="utf-8"))
            payload = build_preview_payload(slideir, job_id="abc")
            self.assertTrue(payload["pages"][0]["baked_background"])
            self.assertIn("width=120", payload["pages"][0]["thumbnail_url"])

    def test_atomic_save_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview_edits.json"
            path.write_text('{"pages":[]}\n', encoding="utf-8")
            save_preview_edits(
                path,
                {"pages": [{"page": 1, "excluded_ids": ["t1"], "role_overrides": {}}]},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pages"][0]["excluded_ids"], ["t1"])
            leftovers = list(Path(tmp).glob(".preview_edits.json.*.tmp"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
