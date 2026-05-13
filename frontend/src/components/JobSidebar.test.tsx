import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { JobSidebar } from "./JobSidebar";

describe("JobSidebar", () => {
  it("lets users adjust detect confidence threshold before running OCR detect", () => {
    const onDetectConfidenceThresholdChange = vi.fn();

    render(
      <JobSidebar
        apiBase=""
        busyAction={null}
        convertResult={null}
        detectConfidenceThreshold={0.75}
        file={new File(["pdf"], "sample.pdf", { type: "application/pdf" })}
        isBusy={false}
        job={{
          job_id: "job_1",
          status: "uploaded",
          original_filename: "sample.pdf",
          page_count: 1,
          created_at: "2026-05-13T00:00:00Z",
          updated_at: "2026-05-13T00:00:00Z",
        }}
        onConvert={vi.fn()}
        onCreateJob={vi.fn()}
        onDetectConfidenceThresholdChange={onDetectConfidenceThresholdChange}
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
});