import type { OverlayVisibility, PreviewBlockRole, PreviewElementRef, PreviewPage } from "../types";

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
            </>
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
