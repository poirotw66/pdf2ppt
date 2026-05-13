import { useState } from "react";

import { InspectorPanel } from "./components/InspectorPanel";
import { JobSidebar } from "./components/JobSidebar";
import { PreviewEditor } from "./components/PreviewEditor";
import { useBoxEditorState } from "./hooks/useBoxEditorState";
import { usePdf2PptApi } from "./hooks/usePdf2PptApi";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const api = usePdf2PptApi();
  const editor = useBoxEditorState({ setStatusText: api.setStatusText });

  async function handleCreateJob() {
    if (!file) {
      api.setStatusText("Select a PDF before creating a job.");
      return;
    }
    const payload = await api.createJob(file);
    if (payload) {
      editor.resetPages();
    }
  }

  async function handleRunDetect() {
    if (!api.job) {
      api.setStatusText("Create a job before running detect.");
      return;
    }
    const payload = await api.runDetect(api.job.job_id);
    if (payload) {
      editor.loadDetectedPages(payload.pages);
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
    await api.convertJob(api.job.job_id);
  }

  return (
    <div className="app-shell">
      <JobSidebar
        apiBase={api.apiBase}
        busyAction={api.busyAction}
        convertResult={api.convertResult}
        file={file}
        isBusy={api.isBusy}
        job={api.job}
        onConvert={handleConvert}
        onCreateJob={handleCreateJob}
        onFileChange={setFile}
        onRunDetect={handleRunDetect}
        onSaveBoxes={() => {
          void handleSaveBoxes();
        }}
        onSelectPage={editor.selectPage}
        pages={editor.pages}
        selectedPageIndex={editor.selectedPageIndex}
        statusText={api.statusText}
      />

      <main className="workspace">
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
      </main>
    </div>
  );
}