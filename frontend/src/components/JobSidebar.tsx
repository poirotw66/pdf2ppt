import {
  INPAINT_ENGINE_OPTIONS,
  LAYOUT_PLANNER_OPTIONS,
  type InpaintEngine,
  type LayoutPlannerOption,
} from "../config";
import type { JobResponse, PagePayload, WorkflowMode } from "../types";
import { UPLOAD_ACCEPT } from "../utils/upload";

type JobSidebarProps = {
  busyAction: string | null;
  convertResult: { page_count: number } | null;
  detectConfidenceThreshold: number;
  inpaintEngine: InpaintEngine;
  layoutPlanner: LayoutPlannerOption;
  file: File | null;
  isBusy: boolean;
  job: JobResponse | null;
  workflowMode: WorkflowMode;
  onWorkflowModeChange: (mode: WorkflowMode) => void;
  onConvert: () => void;
  onBuildPreview: () => void;
  onConvertFromPreview: () => void;
  onDetectConfidenceThresholdChange: (value: number) => void;
  onInpaintEngineChange: (value: InpaintEngine) => void;
  onLayoutPlannerChange: (value: LayoutPlannerOption) => void;
  onFileChange: (file: File | null) => void;
  onRunDetect: () => void;
  onSaveBoxes: () => void;
  onSelectPage: (index: number) => void;
  pages: PagePayload[];
  previewPageCount: number;
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
  layoutPlanner,
  file,
  isBusy,
  job,
  workflowMode,
  onWorkflowModeChange,
  onConvert,
  onBuildPreview,
  onConvertFromPreview,
  onDetectConfidenceThresholdChange,
  onInpaintEngineChange,
  onLayoutPlannerChange,
  onFileChange,
  onRunDetect,
  onSaveBoxes,
  onSelectPage,
  pages,
  previewPageCount,
  selectedPageIndex,
  statusText,
}: JobSidebarProps) {
  const hasFile = file !== null;
  const hasJob = job !== null;
  const hasPages = pages.length > 0;
  const hasPreview = previewPageCount > 0;
  const isPreviewMode = workflowMode === "preview";

  return (
    <aside className="sidebar">
      <div className="panel">
        <h1>pdf2ppt Box Editor</h1>
        <p className="muted">Move through the pipeline from upload to conversion without losing track of the current step.</p>

        <div className="mode-toggle" role="group" aria-label="Workflow mode">
          <button
            type="button"
            className={!isPreviewMode ? "tool-chip active" : "tool-chip"}
            onClick={() => onWorkflowModeChange("ocr")}
            data-testid="mode-ocr"
          >
            OCR Review
          </button>
          <button
            type="button"
            className={isPreviewMode ? "tool-chip active" : "tool-chip"}
            onClick={() => onWorkflowModeChange("preview")}
            data-testid="mode-preview"
          >
            Preview before convert
          </button>
        </div>

        <div className="workflow-steps" aria-label="Workflow steps">
          <section className={hasJob ? "workflow-step complete" : "workflow-step active"}>
            <div className="workflow-step-header">
              <span className="workflow-step-index">1</span>
              <div>
                <strong>Upload Source</strong>
                <p className="muted">Pick a PDF or a single PNG/JPG image and the app will create a new job automatically.</p>
              </div>
            </div>
            <input
              type="file"
              accept={UPLOAD_ACCEPT}
              aria-label="Upload source file"
              onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
            />
            <div className="panel-tip">
              <strong>{hasFile ? file.name : "No source file selected yet"}</strong>
              <span className="muted">
                {hasFile ? (busyAction === "Creating job..." ? "Uploading now and creating a fresh job." : "A new job is created immediately after selection.") : "PDF, PNG, and JPG files are accepted."}
              </span>
            </div>
          </section>

          {isPreviewMode ? (
            <section className={hasJob ? "workflow-step active" : "workflow-step"}>
              <div className="workflow-step-header">
                <span className="workflow-step-index">2</span>
                <div>
                  <strong>Analyze &amp; Preview</strong>
                  <p className="muted">Build SlideIR overlays (planner off). Edit exclusions/roles, then convert with from-ir.</p>
                </div>
              </div>
              <div className="workflow-actions">
                <button onClick={onBuildPreview} disabled={isBusy || !hasJob} data-testid="build-preview">
                  {busyAction === "Building SlideIR preview..." ? "Building Preview..." : "Build Preview"}
                </button>
                <button
                  onClick={onConvertFromPreview}
                  disabled={isBusy || !hasJob || !hasPreview}
                  data-testid="convert-from-preview"
                >
                  {busyAction === "Converting from SlideIR..." ? "Converting..." : "Convert"}
                </button>
              </div>
              <span className="muted step-hint">
                {hasPreview
                  ? `${previewPageCount} page(s) ready. Toggle overlays, exclude elements, then Convert.`
                  : "Upload a PDF, then Build Preview. No Gemini key required."}
              </span>
            </section>
          ) : (
            <>
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
                  {hasJob ? "Run detection once the upload job is ready." : "Upload a PDF or image first to auto-create a job and unlock OCR detection."}
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
                {(inpaintEngine === "generative-openai" || inpaintEngine === "generative-gemini") && (
                  <span className="muted step-hint" data-testid="generative-inpaint-hint">
                    Generative inpaint is opt-in and non-deterministic. Set OPENAI_API_KEY or
                    GEMINI_API_KEY on the API server. Without a key, conversion falls back to local
                    inpainting. Use Convert to regenerate backgrounds for the current review pass.
                  </span>
                )}
                <label className="control-field">
                  <span>Layout Planner (experimental)</span>
                  <select
                    data-testid="layout-planner-select"
                    value={layoutPlanner}
                    onChange={(event) => onLayoutPlannerChange(event.target.value as LayoutPlannerOption)}
                  >
                    {LAYOUT_PLANNER_OPTIONS.map((option) => (
                      <option key={option.value || "off"} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                {layoutPlanner === "gemini" && (
                  <span className="muted step-hint" data-testid="layout-planner-hint">
                    VLM layout planner improves text roles, reading order, and paragraph grouping.
                    Set GEMINI_API_KEY on the API server. Without a key, each page falls back to
                    geometric heuristics. Output is structured JSON only (never pixels).
                  </span>
                )}
                <div className="workflow-actions">
                  <button onClick={onSaveBoxes} disabled={isBusy || !hasJob || !hasPages}>Save Boxes</button>
                  <button onClick={onConvert} disabled={isBusy || !hasJob || !hasPages}>Convert PPTX</button>
                </div>
                <span className="muted step-hint">
                  {hasPages ? "Pick a page, adjust boxes, then save before conversion." : "Run OCR detection to load editable page previews."}
                </span>
              </section>
            </>
          )}
        </div>

        <div className="status-card">
          <strong>Status</strong>
          <span>{statusText}</span>
          {busyAction ? <span className="busy-indicator">{busyAction}</span> : null}
          {job ? <span className="muted">Job: {job.job_id}</span> : null}
          {job && !isPreviewMode ? <span className="muted">Pages detected: {pages.length} / {job.page_count}</span> : null}
          {job && isPreviewMode ? <span className="muted">Preview pages: {previewPageCount} / {job.page_count}</span> : null}
        </div>
        {convertResult && job ? (
          <div className="download-card">
            <strong>Downloads</strong>
            <a href={`${apiBase}/jobs/${job.job_id}/output.pptx`} target="_blank" rel="noreferrer">Download PPTX</a>
            <a href={`${apiBase}/jobs/${job.job_id}/report.json`} target="_blank" rel="noreferrer">Download Report</a>
          </div>
        ) : null}
      </div>

      {!isPreviewMode ? (
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
      ) : null}
    </aside>
  );
}
