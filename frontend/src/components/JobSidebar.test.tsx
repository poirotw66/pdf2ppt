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

afterEach(() => {
  cleanup();
});

describe("JobSidebar", () => {
  it("lets users adjust detect confidence threshold before running OCR detect", () => {
    const onDetectConfidenceThresholdChange = vi.fn();

    render(
      <JobSidebar
        apiBase=""
        busyAction={null}
        convertResult={null}
        detectConfidenceThreshold={0.75}
        inpaintEngine="opencv-fast"
        file={new File(["pdf"], "sample.pdf", { type: "application/pdf" })}
        isBusy={false}
        job={defaultJob}
        onConvert={vi.fn()}
        onCreateJob={vi.fn()}
        onDetectConfidenceThresholdChange={onDetectConfidenceThresholdChange}
        onInpaintEngineChange={vi.fn()}
        onFileChange={vi.fn()}
        onRunDetect={vi.fn()}
        onSaveBoxes={vi.fn()}
        onSelectPage={vi.fn()}
        pages={[]}
        selectedPageIndex={0}
        statusText="ready"
      />,
    );

    fireEvent.change(screen.getByLabelText("Detect Confidence Threshold"), { target: { value: "0.61" } });

    expect(onDetectConfidenceThresholdChange).toHaveBeenCalledWith(0.61);
  });

  it("shows inpaint engine choices for conversion", () => {
    render(
      <JobSidebar
        apiBase=""
        busyAction={null}
        convertResult={null}
        detectConfidenceThreshold={0.75}
        inpaintEngine="lama-pytorch"
        file={new File(["pdf"], "sample.pdf", { type: "application/pdf" })}
        isBusy={false}
        job={defaultJob}
        onConvert={vi.fn()}
        onCreateJob={vi.fn()}
        onDetectConfidenceThresholdChange={vi.fn()}
        onInpaintEngineChange={vi.fn()}
        onFileChange={vi.fn()}
        onRunDetect={vi.fn()}
        onSaveBoxes={vi.fn()}
        onSelectPage={vi.fn()}
        pages={[]}
        selectedPageIndex={0}
        statusText="ready"
      />,
    );

    const inpaintSelects = screen.getAllByTestId("inpaint-engine-select") as HTMLSelectElement[];
    expect(inpaintSelects.some((select) => select.value === "lama-pytorch")).toBe(true);
    expect(Array.from(inpaintSelects[0].options).map((option) => option.value)).toEqual([
      "opencv-fast",
      "auto",
      "white-box",
      "lama-onnx-cuda",
      "lama-pytorch",
    ]);
  });

  it("shows workflow guidance and keeps later actions disabled until prerequisites exist", () => {
    render(
      <JobSidebar
        apiBase=""
        busyAction={null}
        convertResult={null}
        detectConfidenceThreshold={0.75}
        inpaintEngine="opencv-fast"
        file={null}
        isBusy={false}
        job={null}
        onConvert={vi.fn()}
        onCreateJob={vi.fn()}
        onDetectConfidenceThresholdChange={vi.fn()}
        onInpaintEngineChange={vi.fn()}
        onFileChange={vi.fn()}
        onRunDetect={vi.fn()}
        onSaveBoxes={vi.fn()}
        onSelectPage={vi.fn()}
        pages={[]}
        selectedPageIndex={0}
        statusText="idle"
      />,
    );

    expect(screen.getByText("No PDF selected yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create Job" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Run OCR Detect" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Convert PPTX" })).toBeDisabled();
    expect(screen.getByText("Detected pages will appear here after OCR runs.")).toBeInTheDocument();
  });
});