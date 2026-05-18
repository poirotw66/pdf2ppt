import { INPAINT_ENGINE_OPTIONS, type InpaintEngine } from "../config";
import type { JobResponse, PagePayload } from "../types";

type JobSidebarProps = {
  busyAction: string | null;
  convertResult: { page_count: number } | null;
  detectConfidenceThreshold: number;
  inpaintEngine: InpaintEngine;
  file: File | null;
  isBusy: boolean;
  job: JobResponse | null;
  onConvert: () => void;
  onDetectConfidenceThresholdChange: (value: number) => void;
  onInpaintEngineChange: (value: InpaintEngine) => void;
  onFileChange: (file: File | null) => void;
  onRunDetect: () => void;
  onSaveBoxes: () => void;
  onSelectPage: (index: number) => void;
  pages: PagePayload[];
  selectedPageIndex: number;
  statusText: string;
  apiBase: string;
};

export function JobSidebar({
  apiBase,
  busyAction,
  convertResult,
  detectConfidenceThreshold,
  inpaintEngine,
  file,
  isBusy,
  job,
  onConvert,
  onDetectConfidenceThresholdChange,
  onInpaintEngineChange,
  onFileChange,
  onRunDetect,
  onSaveBoxes,
  onSelectPage,
  pages,
  selectedPageIndex,
  statusText,
}: JobSidebarProps) {
  const hasFile = file !== null;
  const hasJob = job !== null;
  const hasPages = pages.length > 0;

  return (
    <aside className="sidebar">
      <div className="panel">
        <h1>pdf2ppt Box Editor</h1>
        <p className="muted">Move through the pipeline from upload to conversion without losing track of the current step.</p>

        <div className="workflow-steps" aria-label="Workflow steps">
          <section className={hasJob ? "workflow-step complete" : "workflow-step active"}>
            <div className="workflow-step-header">
              <span className="workflow-step-index">1</span>
              <div>
                <strong>Upload PDF</strong>
                <p className="muted">Pick a source file and the app will create a new job automatically.</p>
              </div>
            </div>
            <input
              type="file"
              accept="application/pdf"
              aria-label="Upload PDF"
              onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
            />
            <div className="panel-tip">
              <strong>{hasFile ? file.name : "No PDF selected yet"}</strong>
              <span className="muted">
                {hasFile ? (busyAction === "Creating job..." ? "Uploading now and creating a fresh job." : "A new job is created immediately after selection.") : "Only PDF files are accepted."}
              </span>
            </div>
          </section>

          <section className={hasJob ? "workflow-step active" : "workflow-step"}>
            <div className="workflow-step-header">
              <span className="workflow-step-index">2</span>
              <div>
                <strong>Detect OCR Regions</strong>
                <p className="muted">Tune confidence, then generate editable boxes.</p>
              </div>
            </div>
            <label className="control-field">
              <span>Detect Confidence Threshold</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                aria-label="Detect Confidence Threshold"
                value={detectConfidenceThreshold.toFixed(2)}
                onChange={(event) => onDetectConfidenceThresholdChange(Number.parseFloat(event.target.value))}
              />
            </label>
            <button onClick={onRunDetect} disabled={isBusy || !hasJob}>
              {busyAction === "Running OCR detect..." ? "Running OCR Detect..." : "Run OCR Detect"}
            </button>
            <span className="muted step-hint">
              {hasJob ? "Run detection once the upload job is ready." : "Upload a PDF first to auto-create a job and unlock OCR detection."}
            </span>
          </section>

          <section className={hasPages ? "workflow-step active" : "workflow-step"}>
            <div className="workflow-step-header">
              <span className="workflow-step-index">3</span>
              <div>
                <strong>Review And Convert</strong>
                <p className="muted">Save edited boxes, then export the PPTX.</p>
              </div>
            </div>
            <label className="control-field">
              <span>Inpaint Engine</span>
              <select
                data-testid="inpaint-engine-select"
                value={inpaintEngine}
                onChange={(event) => onInpaintEngineChange(event.target.value as InpaintEngine)}
              >
                {INPAINT_ENGINE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="workflow-actions">
              <button onClick={onSaveBoxes} disabled={isBusy || !hasJob || !hasPages}>Save Boxes</button>
              <button onClick={onConvert} disabled={isBusy || !hasJob || !hasPages}>Convert PPTX</button>
            </div>
            <span className="muted step-hint">
              {hasPages ? "Pick a page, adjust boxes, then save before conversion." : "Run OCR detection to load editable page previews."}
            </span>
          </section>
        </div>

        <div className="status-card">
          <strong>Status</strong>
          <span>{statusText}</span>
          {busyAction ? <span className="busy-indicator">{busyAction}</span> : null}
          {job ? <span className="muted">Job: {job.job_id}</span> : null}
          {job ? <span className="muted">Pages detected: {pages.length} / {job.page_count}</span> : null}
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
        {pages.length > 0 ? (
          <div className="page-list">
            {pages.map((page, index) => (
              <button
                key={page.page}
                className={index === selectedPageIndex ? "page-chip active" : "page-chip"}
                onClick={() => onSelectPage(index)}
              >
                <span>Page {page.page}</span>
                <small>{page.boxes.length} boxes</small>
              </button>
            ))}
          </div>
        ) : (
          <div className="empty-state compact-empty-state">Detected pages will appear here after OCR runs.</div>
        )}
      </div>
    </aside>
  );
}