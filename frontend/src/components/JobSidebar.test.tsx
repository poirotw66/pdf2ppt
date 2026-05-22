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
      "lama-pytorch-hybrid",
      "lama-pytorch",
    ]);
    expect(inpaintSelects[0].value).toBe("lama-pytorch");
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

    expect(screen.getByText("No source file selected yet")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create Job" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run OCR Detect" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Convert PPTX" })).toBeDisabled();
    expect(screen.getByText("Detected pages will appear here after OCR runs.")).toBeInTheDocument();
  });

  it("passes the selected file upward immediately", () => {
    const onFileChange = vi.fn();

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
        onDetectConfidenceThresholdChange={vi.fn()}
        onInpaintEngineChange={vi.fn()}
        onFileChange={onFileChange}
        onRunDetect={vi.fn()}
        onSaveBoxes={vi.fn()}
        onSelectPage={vi.fn()}
        pages={[]}
        selectedPageIndex={0}
        statusText="idle"
      />,
    );

    const fileInput = screen.getByLabelText(/upload source file/i) as HTMLInputElement | null;
    const file = new File(["pdf"], "deck.pdf", { type: "application/pdf" });

    fireEvent.change(fileInput ?? screen.getByRole("textbox", { hidden: true }), { target: { files: [file] } });

    expect(onFileChange).toHaveBeenCalledWith(file);
  });
});