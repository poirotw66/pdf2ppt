import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { JobSidebar } from "./JobSidebar";

const defaultJob = {
  job_id: "job_1",
  status: "uploaded",
  original_filename: "sample.pdf",
  page_count: 1,
  created_at: "2026-05-13T00:00:00Z",
  updated_at: "2026-05-13T00:00:00Z",
} as const;

function renderSidebar(overrides: Partial<Parameters<typeof JobSidebar>[0]> = {}) {
  const props = {
    apiBase: "",
    busyAction: null,
    convertResult: null,
    detectConfidenceThreshold: 0.75,
    inpaintEngine: "opencv-fast" as const,
    layoutPlanner: "" as const,
    file: new File(["pdf"], "sample.pdf", { type: "application/pdf" }),
    isBusy: false,
    job: defaultJob,
    workflowMode: "ocr" as const,
    onWorkflowModeChange: vi.fn(),
    onConvert: vi.fn(),
    onBuildPreview: vi.fn(),
    onConvertFromPreview: vi.fn(),
    onDetectConfidenceThresholdChange: vi.fn(),
    onInpaintEngineChange: vi.fn(),
    onLayoutPlannerChange: vi.fn(),
    onFileChange: vi.fn(),
    onRunDetect: vi.fn(),
    onSaveBoxes: vi.fn(),
    onSelectPage: vi.fn(),
    pages: [],
    previewPageCount: 0,
    selectedPageIndex: 0,
    statusText: "ready",
    ...overrides,
  };
  return render(<JobSidebar {...props} />);
}

afterEach(() => {
  cleanup();
});

describe("JobSidebar", () => {
  it("lets users adjust detect confidence threshold before running OCR detect", () => {
    const onDetectConfidenceThresholdChange = vi.fn();
    renderSidebar({ onDetectConfidenceThresholdChange });

    fireEvent.change(screen.getByLabelText("Detect Confidence Threshold"), { target: { value: "0.61" } });

    expect(onDetectConfidenceThresholdChange).toHaveBeenCalledWith(0.61);
  });

  it("shows inpaint engine choices for conversion", () => {
    renderSidebar({ inpaintEngine: "lama-pytorch" });

    const inpaintSelects = screen.getAllByTestId("inpaint-engine-select") as HTMLSelectElement[];
    expect(inpaintSelects.some((select) => select.value === "lama-pytorch")).toBe(true);
    expect(Array.from(inpaintSelects[0].options).map((option) => option.value)).toEqual([
      "opencv-fast",
      "auto",
      "white-box",
      "lama-pytorch-hybrid",
      "lama-pytorch",
      "generative-openai",
      "generative-gemini",
    ]);
    expect(inpaintSelects[0].value).toBe("lama-pytorch");
  });

  it("shows workflow guidance and keeps later actions disabled until prerequisites exist", () => {
    renderSidebar({ file: null, job: null, statusText: "idle" });

    expect(screen.getByText("No source file selected yet")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Job" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run OCR Detect" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Convert PPTX" })).toBeDisabled();
    expect(screen.getByText("Detected pages will appear here after OCR runs.")).toBeInTheDocument();
  });

  it("passes the selected file upward immediately", () => {
    const onFileChange = vi.fn();
    renderSidebar({ file: null, job: null, statusText: "idle", onFileChange });

    const fileInput = screen.getByLabelText(/upload source file/i) as HTMLInputElement | null;
    const file = new File(["pdf"], "deck.pdf", { type: "application/pdf" });

    fireEvent.change(fileInput ?? screen.getByRole("textbox", { hidden: true }), { target: { files: [file] } });

    expect(onFileChange).toHaveBeenCalledWith(file);
  });

  it("switches into preview-before-convert mode with build and convert actions", () => {
    const onWorkflowModeChange = vi.fn();
    const onBuildPreview = vi.fn();
    const onConvertFromPreview = vi.fn();

    renderSidebar({
      workflowMode: "preview",
      previewPageCount: 2,
      onWorkflowModeChange,
      onBuildPreview,
      onConvertFromPreview,
    });

    expect(screen.getByTestId("build-preview")).toBeEnabled();
    expect(screen.getByTestId("convert-from-preview")).toBeEnabled();
    fireEvent.click(screen.getByTestId("convert-from-preview"));
    expect(onConvertFromPreview).toHaveBeenCalled();

    fireEvent.click(screen.getByTestId("mode-ocr"));
    expect(onWorkflowModeChange).toHaveBeenCalledWith("ocr");
  });
});
