import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect, useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PreviewEditor } from "./PreviewEditor";
import { useBoxEditorState } from "../hooks/useBoxEditorState";
import type { PagePayload } from "../types";

const firstPage: PagePayload = {
  page: 1,
  image_url: "/jobs/demo/pages/1.png",
  width: 320,
  height: 240,
  boxes: [
    {
      id: "box_1",
      source: "ocr-auto",
      bbox: [10, 12, 80, 48],
      text: "alpha",
      confidence: 0.91,
    },
  ],
};

const secondPage: PagePayload = {
  page: 2,
  image_url: "/jobs/demo/pages/2.png",
  width: 320,
  height: 240,
  boxes: [
    {
      id: "box_2",
      source: "ocr-auto",
      bbox: [20, 24, 100, 70],
      text: "beta",
      confidence: 0.95,
    },
  ],
};

describe("box editor interactions", () => {
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("creates and deletes a box through editor interactions", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-13T12:00:00Z"));

    const { container } = render(<EditorHarness initialPages={[firstPage]} />);
    const canvas = expectCanvas(container);

    fireEvent.click(screen.getByRole("button", { name: "Create Box" }));
    fireEvent.mouseDown(canvas, { clientX: 120, clientY: 80 });
    fireEvent.mouseMove(canvas, { clientX: 180, clientY: 130 });
    fireEvent.mouseUp(canvas, { clientX: 180, clientY: 130 });

    expect(screen.getByTestId("box-count")).toHaveTextContent("2");
    expect(screen.getByTestId("selected-box")).toHaveTextContent(/^user_1_\d+$/);

    fireEvent.click(screen.getByRole("button", { name: "Delete Selected" }));

    expect(screen.getByTestId("box-count")).toHaveTextContent("1");
    expect(screen.getByTestId("selected-box")).toHaveTextContent("");
  });

  it("selects a box and switches pages while keeping selection in sync", () => {
    render(<EditorHarness initialPages={[firstPage, secondPage]} />);

    fireEvent.click(screen.getAllByText("alpha")[0]);
    expect(screen.getByTestId("selected-box")).toHaveTextContent("box_1");
    expect(screen.getByTestId("current-page")).toHaveTextContent("1");

    fireEvent.click(screen.getByRole("button", { name: "Go to page 2" }));

    expect(screen.getByTestId("current-page")).toHaveTextContent("2");
    expect(screen.getByTestId("selected-box")).toHaveTextContent("box_2");
    expect(screen.getByTestId("box-count")).toHaveTextContent("1");
  });
});

function EditorHarness({ initialPages }: { initialPages: PagePayload[] }) {
  const [statusText, setStatusText] = useState("ready");
  const editor = useBoxEditorState({ setStatusText });

  useEffect(() => {
    editor.loadDetectedPages(initialPages);
  }, []);

  return (
    <div>
      <button type="button" onClick={() => editor.selectPage(0)}>Go to page 1</button>
      <button type="button" onClick={() => editor.selectPage(1)}>Go to page 2</button>
      <button type="button" onClick={editor.deleteSelectedBox}>Delete Selected</button>
      <div data-testid="status-text">{statusText}</div>
      <div data-testid="current-page">{editor.selectedPage?.page ?? 0}</div>
      <div data-testid="box-count">{editor.selectedPage?.boxes.length ?? 0}</div>
      <div data-testid="selected-box">{editor.selectedBoxId ?? ""}</div>
      <PreviewEditor
        apiBase=""
        dragState={editor.dragState}
        editorRef={editor.editorRef}
        editorTool={editor.editorTool}
        editorViewportRef={editor.editorViewportRef}
        onEditorMouseDown={editor.onEditorMouseDown}
        onEditorMouseMove={editor.onEditorMouseMove}
        onEditorMouseUp={editor.onEditorMouseUp}
        onEditorToolChange={editor.setEditorTool}
        onSelectBox={editor.selectBox}
        onStartMoveBox={editor.startMoveBox}
        onStartResizeBox={editor.startResizeBox}
        onZoomChange={editor.onZoomChange}
        previewScale={editor.previewScale}
        selectedBoxId={editor.selectedBoxId}
        selectedBoxIds={editor.selectedBoxIds}
        selectedPage={editor.selectedPage}
        zoom={editor.zoom}
      />
    </div>
  );
}

function expectCanvas(container: HTMLElement): HTMLDivElement {
  const canvas = container.querySelector(".editor-canvas") as HTMLDivElement | null;
  if (!canvas) {
    throw new Error("editor canvas not found");
  }
  Object.defineProperty(canvas, "offsetWidth", {
    configurable: true,
    value: 320,
  });
  Object.defineProperty(canvas, "offsetHeight", {
    configurable: true,
    value: 240,
  });
  vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
    x: 0,
    y: 0,
    width: 320,
    height: 240,
    top: 0,
    left: 0,
    right: 320,
    bottom: 240,
    toJSON: () => ({}),
  } as DOMRect);
  return canvas;
}