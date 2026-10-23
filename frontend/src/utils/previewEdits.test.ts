import { describe, expect, it } from "vitest";

import type { PreviewEdits, PreviewPage } from "../types";
import {
  applyEditsToPreviewPage,
  bboxToPercentStyle,
  getPageEdits,
  setBlockRole,
  setRasterFallback,
  toggleExcluded,
} from "./previewEdits";

const emptyEdits: PreviewEdits = { pages: [] };

const samplePage: PreviewPage = {
  page: 1,
  image_url: "/jobs/x/pages/1.jpg",
  width_pt: 100,
  height_pt: 100,
  page_kind: "digital",
  background_mode: "elements",
  raster_fallback: false,
  text_blocks: [
    {
      id: "t1",
      kind: "text",
      bbox: [10, 10, 40, 30],
      text: "Hello",
      block_role: "body",
      reading_order: 0,
      excluded: false,
    },
  ],
  tables: [],
  images: [{ id: "img_1_0", kind: "image", bbox: [50, 50, 80, 80], excluded: false }],
  shapes: [],
};

describe("previewEdits", () => {
  it("toggles excluded ids and applies them onto a preview page", () => {
    const edits = toggleExcluded(emptyEdits, 1, "t1");
    expect(getPageEdits(edits, 1).excluded_ids).toEqual(["t1"]);
    const applied = applyEditsToPreviewPage(samplePage, edits);
    expect(applied.text_blocks[0]?.excluded).toBe(true);
    const cleared = toggleExcluded(edits, 1, "t1");
    expect(getPageEdits(cleared, 1).excluded_ids).toEqual([]);
  });

  it("stores role overrides only when they differ from the original role", () => {
    const titled = setBlockRole(emptyEdits, 1, "t1", "title", "body");
    expect(getPageEdits(titled, 1).role_overrides).toEqual({ t1: "title" });
    const reverted = setBlockRole(titled, 1, "t1", "body", "body");
    expect(getPageEdits(reverted, 1).role_overrides).toEqual({});
  });

  it("marks raster fallback per page", () => {
    const edits = setRasterFallback(emptyEdits, 1, true);
    expect(getPageEdits(edits, 1).raster_fallback).toBe(true);
    expect(applyEditsToPreviewPage(samplePage, edits).raster_fallback).toBe(true);
  });

  it("maps PDF-point bboxes into percent styles", () => {
    expect(bboxToPercentStyle([10, 20, 60, 70], 100, 100)).toEqual({
      left: "10%",
      top: "20%",
      width: "50%",
      height: "50%",
    });
  });
});
