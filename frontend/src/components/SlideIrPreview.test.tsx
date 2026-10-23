import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { OverlayVisibility, PreviewPage } from "../types";
import { SlideIrPreview } from "./SlideIrPreview";

const overlays: OverlayVisibility = { text: true, tables: true, images: true, shapes: true };

const page: PreviewPage = {
  page: 1,
  image_url: "/jobs/j1/pages/1.jpg",
  width_pt: 200,
  height_pt: 100,
  page_kind: "digital",
  background_mode: "elements",
  raster_fallback: false,
  text_blocks: [
    {
      id: "t1",
      kind: "text",
      bbox: [10, 10, 80, 40],
      text: "Title",
      block_role: "title",
      reading_order: 0,
      excluded: false,
    },
  ],
  tables: [
    {
      id: "tbl1",
      kind: "table",
      bbox: [20, 50, 120, 90],
      row_count: 1,
      col_count: 1,
      cells: [{ row: 0, col: 0, text: "A", bbox: [20, 50, 120, 90] }],
      excluded: false,
    },
  ],
  images: [{ id: "img_1_0", kind: "image", bbox: [140, 20, 180, 60], excluded: true }],
  shapes: [{ id: "s1", kind: "shape", shape_kind: "rect", bbox: [0, 0, 30, 20], excluded: false }],
};

afterEach(() => {
  cleanup();
});

describe("SlideIrPreview", () => {
  it("renders empty state when no page is loaded", () => {
    render(
      <SlideIrPreview
        apiBase=""
        overlayVisibility={overlays}
        page={null}
        selectedElement={null}
        onSelectElement={vi.fn()}
        zoom={1}
      />,
    );
    expect(screen.getByTestId("slideir-preview-empty")).toBeInTheDocument();
    expect(screen.getByText(/No preview yet/i)).toBeInTheDocument();
  });

  it("draws toggleable overlays and reports selection", () => {
    const onSelectElement = vi.fn();
    render(
      <SlideIrPreview
        apiBase=""
        overlayVisibility={overlays}
        page={page}
        selectedElement={{ kind: "text", id: "t1" }}
        onSelectElement={onSelectElement}
        zoom={1}
      />,
    );

    expect(screen.getByTestId("slideir-preview")).toBeInTheDocument();
    fireEvent.click(screen.getByTitle("table tbl1"));
    expect(onSelectElement).toHaveBeenCalledWith({ kind: "table", id: "tbl1" });
    expect(screen.getByTitle("image img_1_0").className).toContain("excluded");
  });

  it("hides layers when overlay visibility is off", () => {
    render(
      <SlideIrPreview
        apiBase=""
        overlayVisibility={{ text: false, tables: false, images: false, shapes: true }}
        page={page}
        selectedElement={null}
        onSelectElement={vi.fn()}
        zoom={1}
      />,
    );
    expect(screen.queryByTitle(/title/i)).not.toBeInTheDocument();
    expect(screen.getByTitle("rect s1")).toBeInTheDocument();
  });

  it("does not draw element overlays when raster fallback is enabled", () => {
    render(
      <SlideIrPreview
        apiBase=""
        overlayVisibility={overlays}
        page={{ ...page, raster_fallback: true }}
        selectedElement={null}
        onSelectElement={vi.fn()}
        zoom={1}
      />,
    );
    expect(screen.queryByTitle("rect s1")).not.toBeInTheDocument();
    expect(screen.queryByTitle("table tbl1")).not.toBeInTheDocument();
  });
});
