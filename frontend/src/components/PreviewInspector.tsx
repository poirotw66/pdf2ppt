import type { OverlayVisibility, PreviewBlockRole, PreviewElementRef, PreviewPage } from "../types";
import { pageHasExistingTitle } from "../utils/previewEdits";

const ROLE_OPTIONS: PreviewBlockRole[] = ["title", "body", "header", "footer", "page_number"];

type PreviewInspectorProps = {
  overlayVisibility: OverlayVisibility;
  onOverlayVisibilityChange: (next: OverlayVisibility) => void;
  page: PreviewPage | null;
  selectedElement: PreviewElementRef;
  onToggleExcluded: () => void;
  onRoleChange: (role: PreviewBlockRole) => void;
  onRasterFallbackChange: (enabled: boolean) => void;
};

export function PreviewInspector({
  overlayVisibility,
  onOverlayVisibilityChange,
  page,
  selectedElement,
  onToggleExcluded,
  onRoleChange,
  onRasterFallbackChange,
}: PreviewInspectorProps) {
  const selected = resolveSelected(page, selectedElement);
  const titleConflict =
    page &&
    selected?.kind === "text" &&
    selected.block_role === "title" &&
    pageHasExistingTitle(page, selected.id);
  const bakedExcludeWarning =
    page?.baked_background && selected && (selected.kind === "shape" || selected.kind === "image");

  return (
    <aside className="inspector-panel" data-testid="preview-inspector">
      <div className="section-header">
        <div className="section-heading">
          <span className="eyebrow">Inspector</span>
          <h2>Preview edits</h2>
          <p className="muted">Toggle overlay layers and apply light edits before Convert.</p>
        </div>
      </div>

      <div className="panel-block">
        <strong>Overlay layers</strong>
        <div className="overlay-toggles" role="group" aria-label="Overlay layers">
          {(
            [
              ["text", "Text blocks"],
              ["tables", "Tables"],
              ["images", "Images"],
              ["shapes", "Shapes"],
            ] as const
          ).map(([key, label]) => (
            <label key={key} className="control-field checkbox-field">
              <input
                type="checkbox"
                checked={overlayVisibility[key]}
                onChange={(event) =>
                  onOverlayVisibilityChange({ ...overlayVisibility, [key]: event.target.checked })
                }
                data-testid={`overlay-toggle-${key}`}
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
      </div>

      {page ? (
        <div className="panel-block">
          <strong>Page {page.page}</strong>
          {page.baked_background ? (
            <span className="muted step-hint" data-testid="baked-background-hint">
              This page uses a baked background raster (overlay/full-page). Excluding shapes or images
              removes native IR objects only — baked pixels may stay visible. If the preview does not
              change, prefer raster fallback for a full-page image.
            </span>
          ) : null}
          <label className="control-field checkbox-field">
            <input
              type="checkbox"
              checked={page.raster_fallback}
              onChange={(event) => onRasterFallbackChange(event.target.checked)}
              data-testid="raster-fallback-toggle"
            />
            <span>Raster fallback (full-page image)</span>
          </label>
          <span className="muted step-hint">
            When enabled, Convert embeds the page image and drops native text / shapes / tables / images for this page.
            Element overlays are hidden while this is on.
          </span>
        </div>
      ) : null}

      {selected ? (
        <div className="panel-block" data-testid="preview-selection">
          <strong>
            {selected.kind} · {selected.id}
          </strong>
          {selected.kind === "text" ? (
            <>
              <p className="muted selection-snippet">{selected.text || "(empty)"}</p>
              <label className="control-field">
                <span>Role</span>
                <select
                  data-testid="preview-role-select"
                  value={normalizeRole(selected.block_role)}
                  onChange={(event) => onRoleChange(event.target.value as PreviewBlockRole)}
                >
                  {ROLE_OPTIONS.map((role) => (
                    <option key={role} value={role}>
                      {role}
                    </option>
                  ))}
                </select>
              </label>
              <span className="muted step-hint">Reading order: {selected.reading_order}</span>
              {titleConflict ? (
                <span className="muted step-hint" data-testid="title-conflict-hint">
                  This page already has another title block. PowerPoint keeps a single title placeholder —
                  the other title stays as a normal text box.
                </span>
              ) : null}
            </>
          ) : null}
          {bakedExcludeWarning ? (
            <span className="muted step-hint" data-testid="baked-exclude-warning">
              Excluding this {selected.kind} may leave it visible if it was painted into the baked
              background. Prefer raster fallback when the preview does not change.
            </span>
          ) : null}
          <div className="inspector-actions">
            <button type="button" onClick={onToggleExcluded} data-testid="toggle-excluded">
              {selected.excluded ? "Include in output" : "Exclude from output"}
            </button>
          </div>
        </div>
      ) : (
        <div className="empty-state inspector-empty-state">
          <strong>Select an overlay</strong>
          <p>Click a text block, table, image, or shape on the preview canvas.</p>
        </div>
      )}
    </aside>
  );
}

function normalizeRole(role: string): PreviewBlockRole {
  if (role === "title" || role === "body" || role === "header" || role === "footer" || role === "page_number") {
    return role;
  }
  return "body";
}

function resolveSelected(page: PreviewPage | null, selected: PreviewElementRef) {
  if (!page || !selected) {
    return null;
  }
  if (selected.kind === "text") {
    const block = page.text_blocks.find((item) => item.id === selected.id);
    return block ? { ...block, kind: "text" as const } : null;
  }
  if (selected.kind === "table") {
    const table = page.tables.find((item) => item.id === selected.id);
    return table ? { ...table, kind: "table" as const } : null;
  }
  if (selected.kind === "image") {
    const image = page.images.find((item) => item.id === selected.id);
    return image ? { ...image, kind: "image" as const } : null;
  }
  const shape = page.shapes.find((item) => item.id === selected.id);
  return shape ? { ...shape, kind: "shape" as const } : null;
}
