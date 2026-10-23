import { useMemo, useState } from "react";

import {
  defaultInpaintEngine,
  defaultLayoutPlanner,
  detectConfidenceThreshold as defaultDetectConfidenceThreshold,
  type InpaintEngine,
  type LayoutPlannerOption,
} from "./config";
import { InspectorPanel } from "./components/InspectorPanel";
import { JobSidebar } from "./components/JobSidebar";
import { PreviewEditor } from "./components/PreviewEditor";
import { PreviewInspector } from "./components/PreviewInspector";
import { SlideIrPreview } from "./components/SlideIrPreview";
import { ThumbnailStrip } from "./components/ThumbnailStrip";
import { useBoxEditorState } from "./hooks/useBoxEditorState";
import { usePdf2PptApi } from "./hooks/usePdf2PptApi";
import type {
  OverlayVisibility,
  PreviewBlockRole,
  PreviewEdits,
  PreviewElementRef,
  PreviewPage,
  WorkflowMode,
} from "./types";
import { applyEditsToPreviewPage, setBlockRole, setRasterFallback, toggleExcluded } from "./utils/previewEdits";
import { isSupportedUpload } from "./utils/upload";

const DEFAULT_OVERLAYS: OverlayVisibility = {
  text: true,
  tables: true,
  images: true,
  shapes: true,
};

export default function App() {
  const [detectConfidenceThreshold, setDetectConfidenceThreshold] = useState(defaultDetectConfidenceThreshold);
  const [inpaintEngine, setInpaintEngine] = useState<InpaintEngine>(defaultInpaintEngine);
  const [layoutPlanner, setLayoutPlanner] = useState<LayoutPlannerOption>(defaultLayoutPlanner);
  const [file, setFile] = useState<File | null>(null);
  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>("ocr");
  const [previewPages, setPreviewPages] = useState<PreviewPage[]>([]);
  const [previewEdits, setPreviewEdits] = useState<PreviewEdits>({ pages: [] });
  const [previewPageIndex, setPreviewPageIndex] = useState(0);
  const [selectedElement, setSelectedElement] = useState<PreviewElementRef>(null);
  const [overlayVisibility, setOverlayVisibility] = useState<OverlayVisibility>(DEFAULT_OVERLAYS);
  const [previewZoom, setPreviewZoom] = useState(1.25);
  const [originalRoles, setOriginalRoles] = useState<Record<string, string>>({});

  const api = usePdf2PptApi();
  const editor = useBoxEditorState({ setStatusText: api.setStatusText });

  const livePreviewPage = useMemo(() => {
    const page = previewPages[previewPageIndex] ?? null;
    if (!page) {
      return null;
    }
    return applyEditsToPreviewPage(page, previewEdits);
  }, [previewPages, previewPageIndex, previewEdits]);

  async function handleFileChange(nextFile: File | null) {
    setFile(nextFile);
    editor.resetPages();
    api.setJob(null);
    resetPreviewState();

    if (!nextFile) {
      api.setStatusText("Upload a PDF or image to start.");
      return;
    }

    if (!isSupportedUpload(nextFile)) {
      api.setStatusText("Only PDF, PNG, and JPG files are supported.");
      setFile(null);
      return;
    }

    const payload = await api.createJob(nextFile);
    if (payload) {
      editor.resetPages();
    }
  }

  function resetPreviewState() {
    setPreviewPages([]);
    setPreviewEdits({ pages: [] });
    setPreviewPageIndex(0);
    setSelectedElement(null);
    setOriginalRoles({});
  }

  async function handleRunDetect() {
    if (!api.job) {
      api.setStatusText("Create a job before running detect.");
      return;
    }
    const payload = await api.runDetect(api.job.job_id, detectConfidenceThreshold);
    if (payload) {
      editor.loadDetectedPages(payload.pages, detectConfidenceThreshold);
    }
  }

  async function handleSaveBoxes() {
    if (!api.job) {
      api.setStatusText("Create a job first.");
      return false;
    }
    return api.saveBoxes(api.job.job_id, editor.pages);
  }

  async function handleConvert() {
    if (!api.job) {
      api.setStatusText("Create a job first.");
      return;
    }
    const saved = await handleSaveBoxes();
    if (!saved) {
      return;
    }
    await api.convertJob(api.job.job_id, inpaintEngine, layoutPlanner);
  }

  async function handleBuildPreview() {
    if (!api.job) {
      api.setStatusText("Create a job first.");
      return;
    }
    // Clear local edits immediately: server also deletes preview_edits.json after re-analysis
    // because element ids are positional and become stale after OCR box changes.
    setPreviewEdits({ pages: [] });
    setSelectedElement(null);
    setOriginalRoles({});
    const payload = await api.buildPreview(api.job.job_id);
    if (!payload) {
      return;
    }
    const roles: Record<string, string> = {};
    for (const page of payload.pages) {
      for (const block of page.text_blocks) {
        roles[`${page.page}:${block.id}`] = block.block_role;
      }
    }
    setOriginalRoles(roles);
    setPreviewPages(payload.pages);
    setPreviewEdits(payload.edits ?? { pages: [] });
    setPreviewPageIndex(0);
    setSelectedElement(null);
  }

  async function handleConvertFromPreview() {
    if (!api.job) {
      api.setStatusText("Create a job first.");
      return;
    }
    if (previewPages.length === 0) {
      api.setStatusText("Build preview before converting.");
      return;
    }
    const saved = await api.savePreviewEdits(api.job.job_id, previewEdits);
    if (!saved) {
      return;
    }
    await api.convertFromIr(api.job.job_id);
  }

  function handleToggleExcluded() {
    if (!livePreviewPage || !selectedElement) {
      return;
    }
    setPreviewEdits((current) => toggleExcluded(current, livePreviewPage.page, selectedElement.id));
  }

  function handleRoleChange(role: PreviewBlockRole) {
    if (!livePreviewPage || !selectedElement || selectedElement.kind !== "text") {
      return;
    }
    const original = originalRoles[`${livePreviewPage.page}:${selectedElement.id}`] ?? "body";
    setPreviewEdits((current) => setBlockRole(current, livePreviewPage.page, selectedElement.id, role, original));
  }

  function handleRasterFallbackChange(enabled: boolean) {
    if (!livePreviewPage) {
      return;
    }
    setPreviewEdits((current) => setRasterFallback(current, livePreviewPage.page, enabled));
  }

  return (
    <div className="app-shell">
      <JobSidebar
        apiBase={api.apiBase}
        busyAction={api.busyAction}
        convertResult={api.convertResult}
        detectConfidenceThreshold={detectConfidenceThreshold}
        inpaintEngine={inpaintEngine}
        layoutPlanner={layoutPlanner}
        file={file}
        isBusy={api.isBusy}
        job={api.job}
        workflowMode={workflowMode}
        onWorkflowModeChange={setWorkflowMode}
        onConvert={handleConvert}
        onBuildPreview={() => {
          void handleBuildPreview();
        }}
        onConvertFromPreview={() => {
          void handleConvertFromPreview();
        }}
        onDetectConfidenceThresholdChange={(value) => {
          if (!Number.isFinite(value)) {
            return;
          }
          setDetectConfidenceThreshold(Math.min(1, Math.max(0, value)));
        }}
        onInpaintEngineChange={setInpaintEngine}
        onLayoutPlannerChange={setLayoutPlanner}
        onFileChange={(nextFile) => {
          void handleFileChange(nextFile);
        }}
        onRunDetect={handleRunDetect}
        onSaveBoxes={() => {
          void handleSaveBoxes();
        }}
        onSelectPage={editor.selectPage}
        pages={editor.pages}
        previewPageCount={previewPages.length}
        selectedPageIndex={editor.selectedPageIndex}
        statusText={api.statusText}
      />

      <main className="workspace">
        {workflowMode === "preview" ? (
          <>
            <div className="preview-workspace">
              <SlideIrPreview
                apiBase={api.apiBase}
                overlayVisibility={overlayVisibility}
                page={livePreviewPage}
                selectedElement={selectedElement}
                onSelectElement={setSelectedElement}
                zoom={previewZoom}
              />
              <ThumbnailStrip
                apiBase={api.apiBase}
                pages={previewPages}
                selectedPageIndex={previewPageIndex}
                onSelectPage={(index) => {
                  setPreviewPageIndex(index);
                  setSelectedElement(null);
                }}
              />
              {previewPages.length > 0 ? (
                <label className="preview-zoom-control">
                  <span>Zoom</span>
                  <input
                    type="range"
                    min="50"
                    max="200"
                    step="10"
                    value={previewZoom * 100}
                    onChange={(event) => setPreviewZoom(Number.parseFloat(event.target.value) / 100)}
                    aria-label="Preview zoom"
                  />
                  <strong>{Math.round(previewZoom * 100)}%</strong>
                </label>
              ) : null}
            </div>
            <PreviewInspector
              overlayVisibility={overlayVisibility}
              onOverlayVisibilityChange={setOverlayVisibility}
              page={livePreviewPage}
              selectedElement={selectedElement}
              onToggleExcluded={handleToggleExcluded}
              onRoleChange={handleRoleChange}
              onRasterFallbackChange={handleRasterFallbackChange}
            />
          </>
        ) : (
          <>
            <PreviewEditor
              apiBase={api.apiBase}
              dragState={editor.dragState}
              editorRef={editor.editorRef}
              editorTool={editor.editorTool}
              editorViewportRef={editor.editorViewportRef}
              onEditorMouseDown={editor.onEditorMouseDown}
              onEditorMouseMove={editor.onEditorMouseMove}
              onEditorMouseUp={editor.onEditorMouseUp}
              onEditorToolChange={editor.setEditorTool}
              onFitToSlide={editor.fitToSlide}
              onSelectBox={editor.selectBox}
              onStartMoveBox={editor.startMoveBox}
              onStartResizeBox={editor.startResizeBox}
              fitZoomPercent={editor.fitZoomPercent}
              onZoomChange={editor.onZoomChange}
              previewScale={editor.previewScale}
              previewZoomPercent={editor.previewZoomPercent}
              selectedBoxId={editor.selectedBoxId}
              selectedBoxIds={editor.selectedBoxIds}
              selectedPage={editor.selectedPage}
              zoom={editor.zoom}
            />

            <InspectorPanel
              boxFilter={editor.boxFilter}
              boxGroup={editor.boxGroup}
              boxSort={editor.boxSort}
              groupedBoxes={editor.groupedBoxes}
              lowConfidenceThreshold={editor.lowConfidenceThreshold}
              onBoxFilterChange={editor.setBoxFilter}
              onBoxListKeyDown={editor.onBoxListKeyDown}
              onDelete={editor.deleteSelectedBox}
              onDuplicate={editor.duplicateSelectedBox}
              onExpand={editor.expandSelectedBox}
              onGroupChange={editor.onGroupChange}
              onNudge={editor.nudgeSelectedBox}
              onSelectBox={editor.selectBox}
              onSortChange={editor.onSortChange}
              onThresholdChange={editor.onThresholdChange}
              onUpdateBBox={editor.updateSelectedBBox}
              onUpdateMeta={editor.updateSelectedBoxMeta}
              onUpdateText={editor.updateSelectedBoxText}
              orderedBoxes={editor.orderedBoxes}
              selectedBox={editor.selectedBox}
              selectedBoxId={editor.selectedBoxId}
              selectedBoxIds={editor.selectedBoxIds}
              selectedPageBoxCount={editor.selectedPage?.boxes.length ?? 0}
            />
          </>
        )}
      </main>
    </div>
  );
}
