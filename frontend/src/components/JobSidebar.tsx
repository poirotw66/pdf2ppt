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
  onCreateJob: () => void;
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
  onCreateJob,
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
  return (
    <aside className="sidebar">
      <div className="panel">
        <h1>pdf2ppt Box Editor</h1>
        <p className="muted">Upload, detect, review OCR boxes, then convert.</p>
        <input
          type="file"
          accept="application/pdf"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
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
        <button onClick={onCreateJob} disabled={isBusy || !file}>{busyAction === "Creating job..." ? "Creating Job..." : "Create Job"}</button>
        <button onClick={onRunDetect} disabled={isBusy || !job}>{busyAction === "Running OCR detect..." ? "Running OCR Detect..." : "Run OCR Detect"}</button>
        <button onClick={onSaveBoxes} disabled={isBusy || !job || pages.length === 0}>Save Boxes</button>
        <button onClick={onConvert} disabled={isBusy || !job || pages.length === 0}>Convert PPTX</button>
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
              onClick={() => onSelectPage(index)}
            >
              <span>Page {page.page}</span>
              <small>{page.boxes.length} boxes</small>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}