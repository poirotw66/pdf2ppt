import type { OverlayVisibility, PreviewElementRef, PreviewPage } from "../types";
import { bboxToPercentStyle } from "../utils/previewEdits";

type SlideIrPreviewProps = {
  apiBase: string;
  overlayVisibility: OverlayVisibility;
  page: PreviewPage | null;
  selectedElement: PreviewElementRef;
  onSelectElement: (element: PreviewElementRef) => void;
  zoom: number;
};

export function SlideIrPreview({
  apiBase,
  overlayVisibility,
  page,
  selectedElement,
  onSelectElement,
  zoom,
}: SlideIrPreviewProps) {
  if (!page) {
    return (
      <section className="editor-panel" data-testid="slideir-preview-empty">
        <div className="section-header">
          <div className="section-heading">
            <span className="eyebrow">SlideIR</span>
            <h2>Preview before convert</h2>
            <p className="muted">
              Analyze the PDF once, inspect detected text / tables / images / shapes, make light edits, then convert
              with from-ir.
            </p>
          </div>
        </div>
        <div className="empty-state editor-empty-state">
          <strong>No preview yet</strong>
          <p>Upload a PDF, then click Build Preview. Overlay data comes from SlideIR — no PPTX render required.</p>
        </div>
      </section>
    );
  }

  const scale = zoom;
  const width = Math.max(page.width_pt, 1) * scale;
  const height = Math.max(page.height_pt, 1) * scale;

  return (
    <section className="editor-panel" data-testid="slideir-preview">
      <div className="section-header">
        <div className="section-heading">
          <span className="eyebrow">SlideIR</span>
          <h2>Preview before convert</h2>
          <p className="muted">
            Toggle overlay layers, exclude elements, change text roles, or mark a page as raster fallback. Convert builds
            PPTX from the edited IR.
          </p>
        </div>
        <div className="header-pills">
          <span className="stat-pill">Page {page.page}</span>
          <span className="stat-pill subtle">{page.page_kind}</span>
          {page.raster_fallback ? <span className="stat-pill warn">Raster fallback</span> : null}
        </div>
      </div>

      <div className="editor-scroll">
        <div className="editor-canvas slideir-canvas" style={{ width, height }} data-testid="slideir-canvas">
          <div
            className="editor-stage"
            style={{ width: page.width_pt, height: page.height_pt, transform: `scale(${scale})` }}
          >
            <img src={`${apiBase}${page.image_url}`} alt={`Page ${page.page}`} draggable={false} />

            {overlayVisibility.shapes
              ? page.shapes.map((shape) => (
                  <button
                    key={shape.id}
                    type="button"
                    className={overlayClass("shape", shape.excluded, selectedElement?.id === shape.id)}
                    style={bboxToPercentStyle(shape.bbox, page.width_pt, page.height_pt)}
                    title={`${shape.shape_kind} ${shape.id}`}
                    onClick={() => onSelectElement({ kind: "shape", id: shape.id })}
                  >
                    <span className="overlay-label">{shape.shape_kind}</span>
                  </button>
                ))
              : null}

            {overlayVisibility.images
              ? page.images.map((image) => (
                  <button
                    key={image.id}
                    type="button"
                    className={overlayClass("image", image.excluded, selectedElement?.id === image.id)}
                    style={bboxToPercentStyle(image.bbox, page.width_pt, page.height_pt)}
                    title={`image ${image.id}`}
                    onClick={() => onSelectElement({ kind: "image", id: image.id })}
                  >
                    <span className="overlay-label">image</span>
                  </button>
                ))
              : null}

            {overlayVisibility.tables
              ? page.tables.flatMap((table) => [
                  <button
                    key={table.id}
                    type="button"
                    className={overlayClass("table", table.excluded, selectedElement?.id === table.id)}
                    style={bboxToPercentStyle(table.bbox, page.width_pt, page.height_pt)}
                    title={`table ${table.id}`}
                    onClick={() => onSelectElement({ kind: "table", id: table.id })}
                  >
                    <span className="overlay-label">
                      table {table.row_count}×{table.col_count}
                    </span>
                  </button>,
                  ...table.cells.map((cell) => (
                    <span
                      key={`${table.id}-cell-${cell.row}-${cell.col}`}
                      className={table.excluded ? "table-cell-overlay excluded" : "table-cell-overlay"}
                      style={bboxToPercentStyle(cell.bbox, page.width_pt, page.height_pt)}
                      aria-hidden="true"
                    />
                  )),
                ])
              : null}

            {overlayVisibility.text
              ? page.text_blocks.map((block) => (
                  <button
                    key={block.id}
                    type="button"
                    className={overlayClass("text", block.excluded, selectedElement?.id === block.id)}
                    style={bboxToPercentStyle(block.bbox, page.width_pt, page.height_pt)}
                    title={`#${block.reading_order} ${block.block_role}: ${block.text}`}
                    onClick={() => onSelectElement({ kind: "text", id: block.id })}
                  >
                    <span className="overlay-label">
                      {block.reading_order}. {block.block_role}
                    </span>
                  </button>
                ))
              : null}
          </div>
        </div>
      </div>
    </section>
  );
}

function overlayClass(kind: string, excluded: boolean, selected: boolean): string {
  return [
    "ir-overlay",
    `ir-overlay-${kind}`,
    excluded ? "excluded" : "",
    selected ? "selected" : "",
  ]
    .filter(Boolean)
    .join(" ");
}
