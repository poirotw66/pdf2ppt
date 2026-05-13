import type { Box, BoxFilter, BoxGroup, BoxGroupEntry, BoxSort } from "../types";

type InspectorPanelProps = {
  boxFilter: BoxFilter;
  boxGroup: BoxGroup;
  boxSort: BoxSort;
  groupedBoxes: BoxGroupEntry[];
  lowConfidenceThreshold: number;
  onBoxFilterChange: (filter: BoxFilter) => void;
  onBoxListKeyDown: (event: React.KeyboardEvent<HTMLButtonElement>, boxId: string) => void;
  onDelete: () => void;
  onDuplicate: () => void;
  onGroupChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
  onSelectBox: (boxId: string) => void;
  onSortChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
  onThresholdChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onUpdateBBox: (index: number, value: string) => void;
  onUpdateMeta: (field: "source" | "confidence", value: string) => void;
  onUpdateText: (text: string) => void;
  orderedBoxes: Box[];
  selectedBox: Box | null;
  selectedBoxId: string | null;
  selectedBoxIds: string[];
  selectedPageBoxCount: number;
  onNudge: (dx: number, dy: number) => void;
  onExpand: (delta: number) => void;
};

export function InspectorPanel({
  boxFilter,
  boxGroup,
  boxSort,
  groupedBoxes,
  lowConfidenceThreshold,
  onBoxFilterChange,
  onBoxListKeyDown,
  onDelete,
  onDuplicate,
  onExpand,
  onGroupChange,
  onNudge,
  onSelectBox,
  onSortChange,
  onThresholdChange,
  onUpdateBBox,
  onUpdateMeta,
  onUpdateText,
  orderedBoxes,
  selectedBox,
  selectedBoxId,
  selectedBoxIds,
  selectedPageBoxCount,
}: InspectorPanelProps) {
  return (
    <section className="inspector-panel">
      <div className="section-header">
        <h2>Selected Box</h2>
        <div className="inspector-actions">
          <button onClick={onDuplicate} disabled={!selectedBox}>Duplicate</button>
          <button onClick={onDelete} disabled={selectedBoxIds.length === 0 && !selectedBox}>Delete</button>
        </div>
      </div>
      {selectedBoxIds.length > 1 ? <div className="multi-select-banner">{selectedBoxIds.length} boxes selected. Delete removes all selected boxes.</div> : null}
      {selectedBox ? (
        <div className="inspector-fields">
          <label>
            <span>Box ID</span>
            <input value={selectedBox.id} readOnly />
          </label>
          <label>
            <span>Source</span>
            <input value={selectedBox.source} onChange={(event) => onUpdateMeta("source", event.target.value)} />
          </label>
          <label>
            <span>Text</span>
            <textarea
              value={selectedBox.text ?? ""}
              placeholder="Leave blank to let convert-time OCR recognize this box."
              onChange={(event) => onUpdateText(event.target.value)}
            />
          </label>
          <label>
            <span>Confidence</span>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={selectedBox.confidence.toFixed(2)}
              onChange={(event) => onUpdateMeta("confidence", event.target.value)}
            />
          </label>
          <div className="bbox-grid">
            {selectedBox.bbox.map((value, index) => (
              <label key={index}>
                <span>{["x0", "y0", "x1", "y1"][index]}</span>
                <input value={value.toFixed(1)} onChange={(event) => onUpdateBBox(index, event.target.value)} />
              </label>
            ))}
          </div>
          <div className="bbox-grid metrics-grid">
            <label>
              <span>Width</span>
              <input value={(selectedBox.bbox[2] - selectedBox.bbox[0]).toFixed(1)} readOnly />
            </label>
            <label>
              <span>Height</span>
              <input value={(selectedBox.bbox[3] - selectedBox.bbox[1]).toFixed(1)} readOnly />
            </label>
          </div>
          <div className="nudge-grid">
            <button onClick={() => onNudge(0, -1)}>Nudge Up</button>
            <button onClick={() => onNudge(-1, 0)}>Nudge Left</button>
            <button onClick={() => onNudge(1, 0)}>Nudge Right</button>
            <button onClick={() => onNudge(0, 1)}>Nudge Down</button>
            <button onClick={() => onExpand(2)}>Expand +2</button>
            <button onClick={() => onExpand(-2)}>Shrink -2</button>
          </div>
        </div>
      ) : (
        <div className="empty-state">Select a box to inspect it.</div>
      )}
      <div className="panel-tip">
        <strong>Editing tips</strong>
        <ul>
          <li>Click empty canvas and drag to create a new box.</li>
          <li>Drag an existing box to reposition it.</li>
          <li>Drag the corner handles to resize a box visually.</li>
          <li>Use arrow keys to nudge and hold Shift for 10 px moves.</li>
          <li>Press Delete or Backspace to remove the selected box.</li>
          <li>Delete bad detections before saving.</li>
          <li>Leave text blank for manual boxes if you want convert-time OCR to fill it.</li>
        </ul>
      </div>

      <div className="panel box-list-panel">
        <div className="section-header">
          <h2>Boxes</h2>
          <span>{orderedBoxes.length}/{selectedPageBoxCount}</span>
        </div>
        <div className="filter-row">
          <button className={boxFilter === "all" ? "filter-chip active" : "filter-chip"} onClick={() => onBoxFilterChange("all")}>All</button>
          <button className={boxFilter === "empty-text" ? "filter-chip active" : "filter-chip"} onClick={() => onBoxFilterChange("empty-text")}>Empty Text</button>
          <button className={boxFilter === "low-confidence" ? "filter-chip active" : "filter-chip"} onClick={() => onBoxFilterChange("low-confidence")}>Low Confidence</button>
        </div>
        <div className="list-controls">
          <label className="control-field">
            <span>Sort</span>
            <select value={boxSort} onChange={onSortChange}>
              <option value="reading-order">Reading Order (Y)</option>
              <option value="confidence-asc">Confidence Low to High</option>
              <option value="confidence-desc">Confidence High to Low</option>
              <option value="source">Source</option>
            </select>
          </label>
          <label className="control-field">
            <span>Group</span>
            <select value={boxGroup} onChange={onGroupChange}>
              <option value="none">None</option>
              <option value="source">Source</option>
            </select>
          </label>
        </div>
        {boxFilter === "low-confidence" ? (
          <label className="threshold-control">
            <span>Threshold</span>
            <input type="number" min="0" max="1" step="0.01" value={lowConfidenceThreshold.toFixed(2)} onChange={onThresholdChange} />
          </label>
        ) : null}
        <div className="shortcut-hint">Use `[` / `]` or `J` / `K` to jump within the current filtered order.</div>
        <div className="box-list">
          {groupedBoxes.map((group) => (
            <div key={group.key} className="box-group">
              {boxGroup !== "none" ? <div className="box-group-title">{group.label} <span>{group.boxes.length}</span></div> : null}
              {group.boxes.map((box) => {
                const isSelected = box.id === selectedBoxId;
                const isEmptyText = !(box.text ?? "").trim();
                return (
                  <button
                    key={box.id}
                    type="button"
                    className={isSelected ? "box-row active" : "box-row"}
                    onClick={() => onSelectBox(box.id)}
                    onKeyDown={(event) => onBoxListKeyDown(event, box.id)}
                  >
                    <div className="box-row-head">
                      <strong>{box.text?.trim() || box.id}</strong>
                      <span>{box.confidence.toFixed(2)}</span>
                    </div>
                    <div className="box-row-meta">
                      <span>{box.source}</span>
                      <span>{Math.round(box.bbox[2] - box.bbox[0])} x {Math.round(box.bbox[3] - box.bbox[1])}</span>
                    </div>
                    <div className="box-row-flags">
                      {isEmptyText ? <span className="flag warning">empty text</span> : null}
                      {box.confidence < lowConfidenceThreshold ? <span className="flag danger">low conf</span> : null}
                    </div>
                  </button>
                );
              })}
            </div>
          ))}
          {orderedBoxes.length === 0 ? <div className="compact-empty">No boxes match the current filter.</div> : null}
        </div>
      </div>
    </section>
  );
}