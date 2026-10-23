import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { OverlayVisibility, PreviewPage } from "../types";
import { PreviewInspector } from "./PreviewInspector";

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
      text: "Hello",
      block_role: "body",
      reading_order: 2,
      excluded: false,
    },
  ],
  tables: [],
  images: [],
  shapes: [],
};

afterEach(() => {
  cleanup();
});

describe("PreviewInspector", () => {
  it("toggles overlay layers and raster fallback", () => {
    const onOverlayVisibilityChange = vi.fn();
    const onRasterFallbackChange = vi.fn();

    render(
      <PreviewInspector
        overlayVisibility={overlays}
        onOverlayVisibilityChange={onOverlayVisibilityChange}
        page={page}
        selectedElement={null}
        onToggleExcluded={vi.fn()}
        onRoleChange={vi.fn()}
        onRasterFallbackChange={onRasterFallbackChange}
      />,
    );

    fireEvent.click(screen.getByTestId("overlay-toggle-shapes"));
    expect(onOverlayVisibilityChange).toHaveBeenCalledWith({ ...overlays, shapes: false });

    fireEvent.click(screen.getByTestId("raster-fallback-toggle"));
    expect(onRasterFallbackChange).toHaveBeenCalledWith(true);
  });

  it("edits selected text role and exclusion", () => {
    const onRoleChange = vi.fn();
    const onToggleExcluded = vi.fn();

    render(
      <PreviewInspector
        overlayVisibility={overlays}
        onOverlayVisibilityChange={vi.fn()}
        page={page}
        selectedElement={{ kind: "text", id: "t1" }}
        onToggleExcluded={onToggleExcluded}
        onRoleChange={onRoleChange}
        onRasterFallbackChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId("preview-selection")).toHaveTextContent("t1");
    fireEvent.change(screen.getByTestId("preview-role-select"), { target: { value: "title" } });
    expect(onRoleChange).toHaveBeenCalledWith("title");
    fireEvent.click(screen.getByTestId("toggle-excluded"));
    expect(onToggleExcluded).toHaveBeenCalled();
  });

  it("warns when setting a second title and when excluding baked shapes", () => {
    const pageWithTitle: PreviewPage = {
      ...page,
      baked_background: true,
      text_blocks: [
        {
          id: "t1",
          kind: "text",
          bbox: [10, 10, 80, 40],
          text: "Hello",
          block_role: "title",
          reading_order: 0,
          excluded: false,
        },
        {
          id: "t2",
          kind: "text",
          bbox: [10, 50, 80, 80],
          text: "Other",
          block_role: "title",
          reading_order: 1,
          excluded: false,
        },
      ],
      shapes: [{ id: "s1", kind: "shape", shape_kind: "rect", bbox: [0, 0, 20, 20], excluded: false }],
    };

    const { rerender } = render(
      <PreviewInspector
        overlayVisibility={overlays}
        onOverlayVisibilityChange={vi.fn()}
        page={pageWithTitle}
        selectedElement={{ kind: "text", id: "t2" }}
        onToggleExcluded={vi.fn()}
        onRoleChange={vi.fn()}
        onRasterFallbackChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("title-conflict-hint")).toBeInTheDocument();

    rerender(
      <PreviewInspector
        overlayVisibility={overlays}
        onOverlayVisibilityChange={vi.fn()}
        page={pageWithTitle}
        selectedElement={{ kind: "shape", id: "s1" }}
        onToggleExcluded={vi.fn()}
        onRoleChange={vi.fn()}
        onRasterFallbackChange={vi.fn()}
      />,
    );
    const warning = screen.getByTestId("baked-exclude-warning");
    expect(warning).toBeInTheDocument();
    expect(warning).toHaveTextContent(/may leave it visible/i);
  });
});
