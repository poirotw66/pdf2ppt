import type { ChangeEvent, MouseEvent, RefObject } from "react";

import type { Box, DragState, EditorTool, PagePayload, ResizeHandle } from "../types";
import { rectToStyle, normalizeRect, resizeHandles } from "../utils/geometry";

type PreviewEditorProps = {
  dragState: DragState;
  editorRef: RefObject<HTMLDivElement>;
  editorTool: EditorTool;
  editorViewportRef: RefObject<HTMLDivElement>;
  onEditorMouseDown: (event: MouseEvent<HTMLDivElement>) => void;
  onEditorMouseMove: (event: MouseEvent<HTMLDivElement>) => void;
  onEditorMouseUp: () => void;
  onEditorToolChange: (tool: EditorTool) => void;
  onSelectBox: (boxId: string) => void;
  onStartMoveBox: (event: MouseEvent<HTMLDivElement>, box: Box) => void;
  onStartResizeBox: (event: MouseEvent<HTMLButtonElement>, box: Box, handle: ResizeHandle) => void;
  onZoomChange: (event: ChangeEvent<HTMLInputElement>) => void;
  previewScale: number;
  selectedBoxId: string | null;
  selectedBoxIds: string[];
  selectedPage: PagePayload | null;
  zoom: number;
  apiBase: string;
};

export function PreviewEditor({
  apiBase,
  dragState,
  editorRef,
  editorTool,
  editorViewportRef,
  onEditorMouseDown,
  onEditorMouseMove,
  onEditorMouseUp,
  onEditorToolChange,
  onSelectBox,
  onStartMoveBox,
  onStartResizeBox,
  onZoomChange,
  previewScale,
  selectedBoxId,
  selectedBoxIds,
  selectedPage,
  zoom,
}: PreviewEditorProps) {
  return (
    <section className="editor-panel">
      <div className="section-header">
        <h2>Preview</h2>
        <div className="preview-toolbar">
          <span>{selectedPage ? `${selectedPage.width} x ${selectedPage.height}` : "No page"}</span>
          <div className="tool-toggle" role="group" aria-label="Preview tool">
            <button
              type="button"
              className={editorTool === "select" ? "tool-chip active" : "tool-chip"}
              onClick={() => onEditorToolChange("select")}
            >
              Select Boxes
            </button>
            <button
              type="button"
              className={editorTool === "create" ? "tool-chip active" : "tool-chip"}
              onClick={() => onEditorToolChange("create")}
            >
              Create Box
            </button>
          </div>
          <label className="zoom-control">
            <span>Zoom</span>
            <input type="range" min="50" max="200" step="10" value={zoom * 100} onChange={onZoomChange} />
            <strong>{Math.round(zoom * 100)}%</strong>
          </label>
        </div>
      </div>
      {selectedPage ? (
        <div ref={editorViewportRef} className="editor-scroll">
          <div
            ref={editorRef}
            className="editor-canvas"
            style={{ width: selectedPage.width * previewScale, height: selectedPage.height * previewScale }}
            onMouseDown={onEditorMouseDown}
            onMouseMove={onEditorMouseMove}
            onMouseUp={onEditorMouseUp}
            onMouseLeave={onEditorMouseUp}
          >
            <div className="editor-stage" style={{ width: selectedPage.width, height: selectedPage.height, transform: `scale(${previewScale})` }}>
              <img src={`${apiBase}${selectedPage.image_url}`} alt={`Page ${selectedPage.page}`} draggable={false} />
              {selectedPage.boxes.map((box) => {
                const [x0, y0, x1, y1] = box.bbox;
                const isSelected = selectedBoxIds.includes(box.id);
                return (
                  <div
                    key={box.id}
                    className={isSelected ? "ocr-box selected" : "ocr-box"}
                    style={{ left: x0, top: y0, width: x1 - x0, height: y1 - y0 }}
                    onMouseDown={(event) => onStartMoveBox(event, box)}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectBox(box.id);
                    }}
                  >
                    <span className="ocr-box-label">{box.text?.trim() || box.id}</span>
                    {box.id === selectedBoxId ? resizeHandles.map((handle) => (
                      <button
                        key={handle}
                        type="button"
                        className={`resize-handle resize-handle-${handle}`}
                        onMouseDown={(event) => onStartResizeBox(event, box, handle)}
                        aria-label={`Resize ${handle}`}
                      />
                    )) : null}
                  </div>
                );
              })}
              {dragState?.kind === "create" || dragState?.kind === "select" ? (
                <div
                  className={dragState.kind === "select" ? "ocr-box draft select-draft" : "ocr-box draft"}
                  style={rectToStyle(normalizeRect(dragState.startX, dragState.startY, dragState.currentX, dragState.currentY))}
                />
              ) : null}
            </div>
          </div>
        </div>
      ) : (
        <div className="empty-state">Run OCR detect to load page previews.</div>
      )}
    </section>
  );
}