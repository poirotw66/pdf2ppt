import { ChangeEvent, KeyboardEvent as ReactKeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

type JobResponse = {
  job_id: string;
  status: string;
  original_filename: string;
  page_count: number;
  approved_boxes_path?: string | null;
  output_pptx_path?: string | null;
  report_path?: string | null;
};

type Box = {
  id: string;
  source: string;
  bbox: [number, number, number, number];
  polygon?: number[][] | null;
  text?: string | null;
  confidence: number;
};

type PagePayload = {
  page: number;
  image_url: string;
  width: number;
  height: number;
  boxes: Box[];
};

type DetectResponse = {
  job_id: string;
  status: string;
  pages: PagePayload[];
};

type ConvertResponse = {
  job_id: string;
  status: string;
  output_pptx_path: string;
  report_path: string;
  page_count: number;
};

type DragState =
  | { kind: "move"; boxId: string; startX: number; startY: number; origin: [number, number, number, number] }
  | {
      kind: "resize";
      boxId: string;
      handle: ResizeHandle;
      startX: number;
      startY: number;
      origin: [number, number, number, number];
    }
  | { kind: "create"; startX: number; startY: number; currentX: number; currentY: number }
  | null;

type ResizeHandle = "nw" | "ne" | "se" | "sw";
type BoxFilter = "all" | "empty-text" | "low-confidence";
type BoxSort = "reading-order" | "confidence-asc" | "confidence-desc" | "source";
type BoxGroup = "none" | "source";

const apiBase = import.meta.env.VITE_API_BASE_URL ?? "";
const lowConfidenceDefault = 0.85;

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [pages, setPages] = useState<PagePayload[]>([]);
  const [selectedPageIndex, setSelectedPageIndex] = useState(0);
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("Upload a PDF to start.");
  const [isBusy, setIsBusy] = useState(false);
  const [convertResult, setConvertResult] = useState<ConvertResponse | null>(null);
  const [dragState, setDragState] = useState<DragState>(null);
  const [zoom, setZoom] = useState(1);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [boxFilter, setBoxFilter] = useState<BoxFilter>("all");
  const [boxSort, setBoxSort] = useState<BoxSort>("reading-order");
  const [boxGroup, setBoxGroup] = useState<BoxGroup>("none");
  const [lowConfidenceThreshold, setLowConfidenceThreshold] = useState(lowConfidenceDefault);
  const editorRef = useRef<HTMLDivElement | null>(null);
  const editorViewportRef = useRef<HTMLDivElement | null>(null);
  const [editorViewportSize, setEditorViewportSize] = useState({ width: 0, height: 0 });

  const selectedPage = pages[selectedPageIndex] ?? null;
  const selectedBox = useMemo(
    () => selectedPage?.boxes.find((box) => box.id === selectedBoxId) ?? null,
    [selectedBoxId, selectedPage],
  );
  const filteredBoxes = useMemo(() => {
    if (!selectedPage) {
      return [];
    }
    return selectedPage.boxes.filter((box) => {
      if (boxFilter === "empty-text") {
        return !(box.text ?? "").trim();
      }
      if (boxFilter === "low-confidence") {
        return box.confidence < lowConfidenceThreshold;
      }
      return true;
    });
  }, [boxFilter, lowConfidenceThreshold, selectedPage]);
  const orderedBoxes = useMemo(() => sortBoxes(filteredBoxes, boxSort), [boxSort, filteredBoxes]);
  const groupedBoxes = useMemo(() => groupBoxes(orderedBoxes, boxGroup), [boxGroup, orderedBoxes]);
  const fitScale = useMemo(() => {
    if (!selectedPage || editorViewportSize.width <= 0 || editorViewportSize.height <= 0) {
      return 1;
    }
    const widthScale = editorViewportSize.width / selectedPage.width;
    const heightScale = editorViewportSize.height / selectedPage.height;
    return Math.min(widthScale, heightScale);
  }, [editorViewportSize.height, editorViewportSize.width, selectedPage]);
  const previewScale = selectedPage ? fitScale * zoom : 1;

  useEffect(() => {
    function onWindowKeyDown(event: KeyboardEvent) {
      if (!selectedPage || !selectedBoxId) {
        return;
      }
      if (dragState) {
        return;
      }
      if (isEditableTarget(event.target)) {
        return;
      }
      const step = event.shiftKey ? 10 : 1;
      switch (event.key) {
        case "Delete":
        case "Backspace":
          event.preventDefault();
          deleteSelectedBox();
          return;
        case "ArrowUp":
          event.preventDefault();
          nudgeSelectedBox(0, -step);
          return;
        case "ArrowDown":
          event.preventDefault();
          nudgeSelectedBox(0, step);
          return;
        case "ArrowLeft":
          event.preventDefault();
          nudgeSelectedBox(-step, 0);
          return;
        case "ArrowRight":
          event.preventDefault();
          nudgeSelectedBox(step, 0);
          return;
        case "[":
        case "j":
        case "J":
          event.preventDefault();
          selectRelativeBox(-1);
          return;
        case "]":
        case "k":
        case "K":
          event.preventDefault();
          selectRelativeBox(1);
          return;
        default:
          return;
      }
    }

    window.addEventListener("keydown", onWindowKeyDown);
    return () => window.removeEventListener("keydown", onWindowKeyDown);
  }, [dragState, orderedBoxes, selectedBoxId, selectedPage]);

  useEffect(() => {
    if (!editorViewportRef.current) {
      return;
    }
    const element = editorViewportRef.current;
    const updateSize = () => {
      setEditorViewportSize({
        width: Math.max(element.clientWidth - 36, 1),
        height: Math.max(element.clientHeight - 36, 1),
      });
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setDragState(null);
  }, [selectedPageIndex]);

  async function createJob() {
    if (!file) {
      setStatusText("Select a PDF before creating a job.");
      return;
    }
    setIsBusy(true);
    setBusyAction("Creating job...");
    setStatusText(`Uploading ${file.name} and creating job...`);
    setConvertResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${apiBase}/jobs`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as JobResponse;
      setJob(payload);
      setPages([]);
      setSelectedPageIndex(0);
      setSelectedBoxId(null);
      setStatusText(`Job ${payload.job_id} created. Run OCR detect next.`);
    } catch (error) {
      setStatusText(`Create job failed: ${stringifyError(error)}`);
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  async function runDetect() {
    if (!job) {
      setStatusText("Create a job before running detect.");
      return;
    }
    setIsBusy(true);
    setBusyAction("Running OCR detect...");
    setStatusText(`Running OCR detect for job ${job.job_id}...`);
    try {
      const response = await fetch(`${apiBase}/jobs/${job.job_id}/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dpi: 144 }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as DetectResponse;
      setPages(payload.pages);
      setSelectedPageIndex(0);
      setSelectedBoxId(payload.pages[0]?.boxes[0]?.id ?? null);
      setStatusText(`Detected ${payload.pages.reduce((count, page) => count + page.boxes.length, 0)} boxes.`);
    } catch (error) {
      setStatusText(`Detect failed: ${stringifyError(error)}`);
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  async function saveBoxes() {
    if (!job || pages.length === 0) {
      setStatusText("Nothing to save yet.");
      return false;
    }
    setIsBusy(true);
    setBusyAction("Saving boxes...");
    setStatusText("Saving approved boxes...");
    try {
      const response = await fetch(`${apiBase}/jobs/${job.job_id}/boxes`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pages: pages.map((page) => ({
            page: page.page,
            width: page.width,
            height: page.height,
            boxes: page.boxes,
          })),
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const refreshed = await fetch(`${apiBase}/jobs/${job.job_id}`);
      if (refreshed.ok) {
        setJob((await refreshed.json()) as JobResponse);
      }
      setStatusText("Approved boxes saved.");
      return true;
    } catch (error) {
      setStatusText(`Save boxes failed: ${stringifyError(error)}`);
      return false;
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  async function convertJob() {
    if (!job) {
      setStatusText("Create a job first.");
      return;
    }
    const saved = await saveBoxes();
    if (!saved) {
      return;
    }
    setIsBusy(true);
    setBusyAction("Converting PPTX...");
    setStatusText(`Converting job ${job.job_id} into PPTX...`);
    try {
      const response = await fetch(`${apiBase}/jobs/${job.job_id}/convert`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ write_debug_artifacts: false }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as ConvertResponse;
      setConvertResult(payload);
      const refreshed = await fetch(`${apiBase}/jobs/${job.job_id}`);
      if (refreshed.ok) {
        setJob((await refreshed.json()) as JobResponse);
      }
      setStatusText(`Conversion finished for ${payload.page_count} page(s).`);
    } catch (error) {
      setStatusText(`Convert failed: ${stringifyError(error)}`);
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  function updateCurrentPage(mutator: (page: PagePayload) => PagePayload) {
    setPages((currentPages) =>
      currentPages.map((page, index) => (index === selectedPageIndex ? mutator(page) : page)),
    );
  }

  function updateSelectedBox(mutator: (box: Box, page: PagePayload) => Box) {
    if (!selectedBoxId) {
      return;
    }
    updateCurrentPage((page) => ({
      ...page,
      boxes: page.boxes.map((box) => (box.id === selectedBoxId ? mutator(box, page) : box)),
    }));
  }

  function deleteSelectedBox() {
    if (!selectedPage || !selectedBoxId) {
      return;
    }
    updateCurrentPage((page) => ({
      ...page,
      boxes: page.boxes.filter((box) => box.id !== selectedBoxId),
    }));
    setSelectedBoxId(null);
  }

  function onEditorMouseDown(event: React.MouseEvent<HTMLDivElement>) {
    if (!selectedPage || !editorRef.current) {
      return;
    }
    const target = event.target as HTMLElement;
    if (target.closest(".ocr-box") || target.closest(".resize-handle")) {
      return;
    }
    const point = getLocalPoint(event, editorRef.current, previewScale);
    setSelectedBoxId(null);
    setDragState({ kind: "create", startX: point.x, startY: point.y, currentX: point.x, currentY: point.y });
  }

  function onEditorMouseMove(event: React.MouseEvent<HTMLDivElement>) {
    if (!dragState || !editorRef.current) {
      return;
    }
    const point = getLocalPoint(event, editorRef.current, previewScale);
    if (dragState.kind === "create") {
      setDragState({ ...dragState, currentX: point.x, currentY: point.y });
      return;
    }
    if (dragState.kind === "move") {
      const dx = point.x - dragState.startX;
      const dy = point.y - dragState.startY;
      updateCurrentPage((page) => ({
        ...page,
        boxes: page.boxes.map((box) => {
          if (box.id !== dragState.boxId) {
            return box;
          }
          const [x0, y0, x1, y1] = dragState.origin;
          const width = x1 - x0;
          const height = y1 - y0;
          const nextX0 = clamp(x0 + dx, 0, page.width - width);
          const nextY0 = clamp(y0 + dy, 0, page.height - height);
          return {
            ...box,
            bbox: [nextX0, nextY0, nextX0 + width, nextY0 + height],
            polygon: rectToPolygon(nextX0, nextY0, nextX0 + width, nextY0 + height),
          };
        }),
      }));
      return;
    }
    if (dragState.kind === "resize") {
      updateCurrentPage((page) => ({
        ...page,
        boxes: page.boxes.map((box) => {
          if (box.id !== dragState.boxId) {
            return box;
          }
          const nextBbox = resizeBBox(dragState.origin, dragState.handle, point.x, point.y, page.width, page.height);
          return {
            ...box,
            bbox: nextBbox,
            polygon: rectToPolygon(...nextBbox),
          };
        }),
      }));
    }
  }

  function onEditorMouseUp() {
    if (!selectedPage || !dragState) {
      setDragState(null);
      return;
    }
    if (dragState.kind === "create") {
      const bbox = normalizeRect(dragState.startX, dragState.startY, dragState.currentX, dragState.currentY);
      const [x0, y0, x1, y1] = bbox;
      if (x1 - x0 >= 8 && y1 - y0 >= 8) {
        const newId = `user_${selectedPage.page}_${Date.now()}`;
        updateCurrentPage((page) => ({
          ...page,
          boxes: [
            ...page.boxes,
            {
              id: newId,
              source: "ocr-user",
              bbox,
              polygon: rectToPolygon(x0, y0, x1, y1),
              text: "",
              confidence: 1,
            },
          ],
        }));
        setSelectedBoxId(newId);
      }
    }
    setDragState(null);
  }

  function startMoveBox(event: React.MouseEvent<HTMLDivElement>, box: Box) {
    event.stopPropagation();
    if (!editorRef.current) {
      return;
    }
    const point = getLocalPoint(event, editorRef.current, previewScale);
    setSelectedBoxId(box.id);
    setDragState({ kind: "move", boxId: box.id, startX: point.x, startY: point.y, origin: box.bbox });
  }

  function startResizeBox(event: React.MouseEvent<HTMLButtonElement>, box: Box, handle: ResizeHandle) {
    event.stopPropagation();
    if (!editorRef.current) {
      return;
    }
    const point = getLocalPoint(event, editorRef.current, previewScale);
    setSelectedBoxId(box.id);
    setDragState({ kind: "resize", boxId: box.id, handle, startX: point.x, startY: point.y, origin: box.bbox });
  }

  function updateSelectedBoxText(text: string) {
    updateSelectedBox((box) => ({ ...box, text }));
  }

  function updateSelectedBoxMeta(field: "source" | "confidence", value: string) {
    updateSelectedBox((box) => {
      if (field === "confidence") {
        const confidence = Number.parseFloat(value);
        return { ...box, confidence: Number.isFinite(confidence) ? clamp(confidence, 0, 1) : box.confidence };
      }
      return { ...box, source: value };
    });
  }

  function updateSelectedBBox(index: number, value: string) {
    updateSelectedBox((box, page) => {
      const parsed = Number.parseFloat(value);
      if (!Number.isFinite(parsed)) {
        return box;
      }
      const next = [...box.bbox] as [number, number, number, number];
      next[index] = parsed;
      const normalized = normalizeAndClampRect(next[0], next[1], next[2], next[3], page.width, page.height);
      return {
        ...box,
        bbox: normalized,
        polygon: rectToPolygon(...normalized),
      };
    });
  }

  function nudgeSelectedBox(dx: number, dy: number) {
    updateSelectedBox((box, page) => {
      const [x0, y0, x1, y1] = box.bbox;
      const width = x1 - x0;
      const height = y1 - y0;
      const nextX0 = clamp(x0 + dx, 0, page.width - width);
      const nextY0 = clamp(y0 + dy, 0, page.height - height);
      const normalized = [nextX0, nextY0, nextX0 + width, nextY0 + height] as [number, number, number, number];
      return {
        ...box,
        bbox: normalized,
        polygon: rectToPolygon(...normalized),
      };
    });
  }

  function expandSelectedBox(delta: number) {
    updateSelectedBox((box, page) => {
      const [x0, y0, x1, y1] = box.bbox;
      const normalized = normalizeAndClampRect(x0 - delta, y0 - delta, x1 + delta, y1 + delta, page.width, page.height);
      return {
        ...box,
        bbox: normalized,
        polygon: rectToPolygon(...normalized),
      };
    });
  }

  function duplicateSelectedBox() {
    if (!selectedBox) {
      return;
    }
    const newId = `${selectedBox.id}_copy_${Date.now()}`;
    const duplicated = {
      ...selectedBox,
      id: newId,
      bbox: [
        selectedBox.bbox[0] + 8,
        selectedBox.bbox[1] + 8,
        selectedBox.bbox[2] + 8,
        selectedBox.bbox[3] + 8,
      ] as [number, number, number, number],
    };
    duplicated.polygon = rectToPolygon(...duplicated.bbox);
    updateCurrentPage((page) => ({
      ...page,
      boxes: [...page.boxes, duplicated],
    }));
    setSelectedBoxId(newId);
  }

  function onZoomChange(event: ChangeEvent<HTMLInputElement>) {
    setZoom(Number.parseInt(event.target.value, 10) / 100);
  }

  function onThresholdChange(event: ChangeEvent<HTMLInputElement>) {
    const next = Number.parseFloat(event.target.value);
    if (!Number.isFinite(next)) {
      return;
    }
    setLowConfidenceThreshold(clamp(next, 0, 1));
  }

  function onSortChange(event: ChangeEvent<HTMLSelectElement>) {
    setBoxSort(event.target.value as BoxSort);
  }

  function onGroupChange(event: ChangeEvent<HTMLSelectElement>) {
    setBoxGroup(event.target.value as BoxGroup);
  }

  function onBoxListKeyDown(event: ReactKeyboardEvent<HTMLButtonElement>, boxId: string) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setSelectedBoxId(boxId);
    }
  }

  function selectRelativeBox(direction: -1 | 1) {
    if (orderedBoxes.length === 0) {
      return;
    }
    const currentIndex = orderedBoxes.findIndex((box) => box.id === selectedBoxId);
    const nextIndex = currentIndex === -1
      ? direction > 0 ? 0 : orderedBoxes.length - 1
      : clamp(currentIndex + direction, 0, orderedBoxes.length - 1);
    setSelectedBoxId(orderedBoxes[nextIndex]?.id ?? null);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="panel">
          <h1>pdf2ppt Box Editor</h1>
          <p className="muted">Upload, detect, review OCR boxes, then convert.</p>
          <input
            type="file"
            accept="application/pdf"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <button onClick={createJob} disabled={isBusy || !file}>{busyAction === "Creating job..." ? "Creating Job..." : "Create Job"}</button>
          <button onClick={runDetect} disabled={isBusy || !job}>{busyAction === "Running OCR detect..." ? "Running OCR Detect..." : "Run OCR Detect"}</button>
          <button onClick={saveBoxes} disabled={isBusy || !job || pages.length === 0}>Save Boxes</button>
          <button onClick={convertJob} disabled={isBusy || !job || pages.length === 0}>Convert PPTX</button>
          <div className="status-card">
            <strong>Status</strong>
            <span>{statusText}</span>
            {busyAction ? <span className="busy-indicator">{busyAction}</span> : null}
            {job ? <span className="muted">Job: {job.job_id}</span> : null}
          </div>
          {convertResult && job ? (
            <div className="download-card">
              <strong>Downloads</strong>
              <a href={`${apiBase}/jobs/${job.job_id}/output.pptx`} target="_blank" rel="noreferrer">Download PPTX</a>
              <a href={`${apiBase}/jobs/${job.job_id}/report.json`} target="_blank" rel="noreferrer">Download Report</a>
            </div>
          ) : null}
        </div>

        <div className="panel page-list-panel">
          <div className="section-header">
            <h2>Pages</h2>
            <span>{pages.length}</span>
          </div>
          <div className="page-list">
            {pages.map((page, index) => (
              <button
                key={page.page}
                className={index === selectedPageIndex ? "page-chip active" : "page-chip"}
                onClick={() => {
                  setSelectedPageIndex(index);
                  setSelectedBoxId(page.boxes[0]?.id ?? null);
                  setStatusText(`Viewing page ${page.page}.`);
                }}
              >
                <span>Page {page.page}</span>
                <small>{page.boxes.length} boxes</small>
              </button>
            ))}
          </div>
        </div>
      </aside>

      <main className="workspace">
        <section className="editor-panel">
          <div className="section-header">
            <h2>Preview</h2>
            <div className="preview-toolbar">
              <span>{selectedPage ? `${selectedPage.width} x ${selectedPage.height}` : "No page"}</span>
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
                <div key={selectedPage.page} className="editor-stage" style={{ width: selectedPage.width, height: selectedPage.height, transform: `scale(${previewScale})` }}>
                  <img key={selectedPage.image_url} src={`${apiBase}${selectedPage.image_url}`} alt={`Page ${selectedPage.page}`} draggable={false} />
                {selectedPage.boxes.map((box) => {
                  const [x0, y0, x1, y1] = box.bbox;
                  return (
                    <div
                      key={box.id}
                      className={box.id === selectedBoxId ? "ocr-box selected" : "ocr-box"}
                      style={{ left: x0, top: y0, width: x1 - x0, height: y1 - y0 }}
                      onMouseDown={(event) => startMoveBox(event, box)}
                      onClick={(event) => {
                        event.stopPropagation();
                        setSelectedBoxId(box.id);
                      }}
                    >
                      <span className="ocr-box-label">{box.text?.trim() || box.id}</span>
                      {box.id === selectedBoxId ? resizeHandles.map((handle) => (
                        <button
                          key={handle}
                          type="button"
                          className={`resize-handle resize-handle-${handle}`}
                          onMouseDown={(event) => startResizeBox(event, box, handle)}
                          aria-label={`Resize ${handle}`}
                        />
                      )) : null}
                    </div>
                  );
                })}
                {dragState?.kind === "create" ? (
                  <div
                    className="ocr-box draft"
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

        <section className="inspector-panel">
          <div className="section-header">
            <h2>Selected Box</h2>
            <div className="inspector-actions">
              <button onClick={duplicateSelectedBox} disabled={!selectedBox}>Duplicate</button>
              <button onClick={deleteSelectedBox} disabled={!selectedBox}>Delete</button>
            </div>
          </div>
          {selectedBox ? (
            <div className="inspector-fields">
              <label>
                <span>Box ID</span>
                <input value={selectedBox.id} readOnly />
              </label>
              <label>
                <span>Source</span>
                <input value={selectedBox.source} onChange={(event) => updateSelectedBoxMeta("source", event.target.value)} />
              </label>
              <label>
                <span>Text</span>
                <textarea
                  value={selectedBox.text ?? ""}
                  placeholder="Leave blank to let convert-time OCR recognize this box."
                  onChange={(event) => updateSelectedBoxText(event.target.value)}
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
                  onChange={(event) => updateSelectedBoxMeta("confidence", event.target.value)}
                />
              </label>
              <div className="bbox-grid">
                {selectedBox.bbox.map((value, index) => (
                  <label key={index}>
                    <span>{["x0", "y0", "x1", "y1"][index]}</span>
                    <input value={value.toFixed(1)} onChange={(event) => updateSelectedBBox(index, event.target.value)} />
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
                <button onClick={() => nudgeSelectedBox(0, -1)}>Nudge Up</button>
                <button onClick={() => nudgeSelectedBox(-1, 0)}>Nudge Left</button>
                <button onClick={() => nudgeSelectedBox(1, 0)}>Nudge Right</button>
                <button onClick={() => nudgeSelectedBox(0, 1)}>Nudge Down</button>
                <button onClick={() => expandSelectedBox(2)}>Expand +2</button>
                <button onClick={() => expandSelectedBox(-2)}>Shrink -2</button>
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
              <span>{orderedBoxes.length}/{selectedPage?.boxes.length ?? 0}</span>
            </div>
            <div className="filter-row">
              <button
                className={boxFilter === "all" ? "filter-chip active" : "filter-chip"}
                onClick={() => setBoxFilter("all")}
              >
                All
              </button>
              <button
                className={boxFilter === "empty-text" ? "filter-chip active" : "filter-chip"}
                onClick={() => setBoxFilter("empty-text")}
              >
                Empty Text
              </button>
              <button
                className={boxFilter === "low-confidence" ? "filter-chip active" : "filter-chip"}
                onClick={() => setBoxFilter("low-confidence")}
              >
                Low Confidence
              </button>
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
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={lowConfidenceThreshold.toFixed(2)}
                  onChange={onThresholdChange}
                />
              </label>
            ) : null}
            <div className="shortcut-hint">Use `[` / `]` or `J` / `K` to jump within the current filtered order.</div>
            {selectedPage ? (
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
                          onClick={() => setSelectedBoxId(box.id)}
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
            ) : (
              <div className="compact-empty">Run OCR detect to populate the box list.</div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function stringifyError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeRect(x0: number, y0: number, x1: number, y1: number): [number, number, number, number] {
  return [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)];
}

function normalizeAndClampRect(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  maxWidth: number,
  maxHeight: number,
): [number, number, number, number] {
  const [nx0, ny0, nx1, ny1] = normalizeRect(
    clamp(x0, 0, maxWidth),
    clamp(y0, 0, maxHeight),
    clamp(x1, 0, maxWidth),
    clamp(y1, 0, maxHeight),
  );
  const minSize = 4;
  return [nx0, ny0, Math.max(nx0 + minSize, nx1), Math.max(ny0 + minSize, ny1)];
}

function rectToPolygon(x0: number, y0: number, x1: number, y1: number): number[][] {
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
}

function rectToStyle([x0, y0, x1, y1]: [number, number, number, number]) {
  return {
    left: x0,
    top: y0,
    width: x1 - x0,
    height: y1 - y0,
  };
}

function getLocalPoint(event: React.MouseEvent<HTMLElement>, element: HTMLDivElement, scale: number) {
  const rect = element.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / (rect.width / element.offsetWidth)) / scale,
    y: ((event.clientY - rect.top) / (rect.height / element.offsetHeight)) / scale,
  };
}

function resizeBBox(
  origin: [number, number, number, number],
  handle: ResizeHandle,
  pointerX: number,
  pointerY: number,
  maxWidth: number,
  maxHeight: number,
): [number, number, number, number] {
  let [x0, y0, x1, y1] = origin;
  if (handle.includes("n")) {
    y0 = pointerY;
  }
  if (handle.includes("s")) {
    y1 = pointerY;
  }
  if (handle.includes("w")) {
    x0 = pointerX;
  }
  if (handle.includes("e")) {
    x1 = pointerX;
  }
  return normalizeAndClampRect(x0, y0, x1, y1, maxWidth, maxHeight);
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  if (target.isContentEditable) {
    return true;
  }
  const tagName = target.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

function sortBoxes(boxes: Box[], sortBy: BoxSort): Box[] {
  const nextBoxes = [...boxes];
  nextBoxes.sort((left, right) => {
    if (sortBy === "confidence-asc") {
      return left.confidence - right.confidence || compareReadingOrder(left, right);
    }
    if (sortBy === "confidence-desc") {
      return right.confidence - left.confidence || compareReadingOrder(left, right);
    }
    if (sortBy === "source") {
      return left.source.localeCompare(right.source) || compareReadingOrder(left, right);
    }
    return compareReadingOrder(left, right);
  });
  return nextBoxes;
}

function compareReadingOrder(left: Box, right: Box): number {
  const yDiff = left.bbox[1] - right.bbox[1];
  if (Math.abs(yDiff) > 6) {
    return yDiff;
  }
  return left.bbox[0] - right.bbox[0];
}

function groupBoxes(boxes: Box[], groupBy: BoxGroup): Array<{ key: string; label: string; boxes: Box[] }> {
  if (groupBy === "none") {
    return [{ key: "all", label: "All Boxes", boxes }];
  }
  const grouped = new Map<string, Box[]>();
  for (const box of boxes) {
    const key = box.source || "unknown";
    const existing = grouped.get(key);
    if (existing) {
      existing.push(box);
    } else {
      grouped.set(key, [box]);
    }
  }
  return Array.from(grouped.entries()).map(([key, groupBoxes]) => ({
    key,
    label: key,
    boxes: groupBoxes,
  }));
}

const resizeHandles: ResizeHandle[] = ["nw", "ne", "se", "sw"];