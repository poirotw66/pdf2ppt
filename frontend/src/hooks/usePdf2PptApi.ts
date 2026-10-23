import { useState } from "react";

import {
  buildConvertRequestPayload,
  detectConfidenceThreshold,
  type InpaintEngine,
} from "../config";
import type {
  ApiErrorResponse,
  ConvertResponse,
  DetectResponse,
  FromIrJobResponse,
  JobResponse,
  PagePayload,
  PreviewEdits,
  PreviewResponse,
} from "../types";

const apiBase = import.meta.env.VITE_API_BASE_URL ?? "";

export function usePdf2PptApi() {
  const [job, setJob] = useState<JobResponse | null>(null);
  const [statusText, setStatusText] = useState("Upload a PDF or image to start.");
  const [isBusy, setIsBusy] = useState(false);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [convertResult, setConvertResult] = useState<ConvertResponse | null>(null);

  async function createJob(file: File): Promise<JobResponse | null> {
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
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as JobResponse;
      setJob(payload);
      setStatusText(`Job ${payload.job_id} created. Run OCR detect next.`);
      return payload;
    } catch (error) {
      setStatusText(`Create job failed: ${stringifyError(error)}`);
      return null;
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  async function runDetect(jobId: string, confidenceThreshold: number = detectConfidenceThreshold): Promise<DetectResponse | null> {
    setIsBusy(true);
    setBusyAction("Running OCR detect...");
    setStatusText(`Running OCR detect for job ${jobId}...`);
    try {
      const response = await fetch(`${apiBase}/jobs/${jobId}/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dpi: 144, confidence_threshold: confidenceThreshold }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as DetectResponse;
      setStatusText(`Detected ${payload.pages.reduce((count, page) => count + page.boxes.length, 0)} boxes.`);
      return payload;
    } catch (error) {
      setStatusText(`Detect failed: ${stringifyError(error)}`);
      return null;
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  async function saveBoxes(jobId: string, pages: PagePayload[]): Promise<boolean> {
    if (pages.length === 0) {
      setStatusText("Nothing to save yet.");
      return false;
    }
    setIsBusy(true);
    setBusyAction("Saving boxes...");
    setStatusText("Saving approved boxes...");
    try {
      const response = await fetch(`${apiBase}/jobs/${jobId}/boxes`, {
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
        throw new Error(await readError(response));
      }
      await refreshJob(jobId);
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

  async function convertJob(
    jobId: string,
    inpaintEngine: InpaintEngine,
    layoutPlanner: import("../config").LayoutPlannerOption = "",
  ): Promise<ConvertResponse | null> {
    setIsBusy(true);
    setBusyAction("Converting PPTX...");
    const lamaHint =
      inpaintEngine === "lama-pytorch" || inpaintEngine === "lama-pytorch-hybrid"
        ? " LaMa may take several minutes on dense slides (first run loads the GPU model)."
        : "";
    setStatusText(`Converting job ${jobId} into PPTX...${lamaHint}`);
    try {
      const response = await fetch(`${apiBase}/jobs/${jobId}/convert`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildConvertRequestPayload(inpaintEngine, false, layoutPlanner)),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as ConvertResponse;
      setConvertResult(payload);
      await refreshJob(jobId);
      setStatusText(`Conversion finished for ${payload.page_count} page(s).`);
      return payload;
    } catch (error) {
      setStatusText(`Convert failed: ${stringifyError(error)}`);
      return null;
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  async function refreshJob(jobId: string): Promise<JobResponse | null> {
    const response = await fetch(`${apiBase}/jobs/${jobId}`);
    if (!response.ok) {
      return null;
    }
    const payload = (await response.json()) as JobResponse;
    setJob(payload);
    return payload;
  }

  async function buildPreview(jobId: string): Promise<PreviewResponse | null> {
    setIsBusy(true);
    setBusyAction("Building SlideIR preview...");
    setStatusText(`Analyzing job ${jobId} for preview (planner off)...`);
    try {
      const response = await fetch(`${apiBase}/jobs/${jobId}/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as PreviewResponse;
      await refreshJob(jobId);
      setStatusText(`Preview ready for ${payload.page_count} page(s).`);
      return payload;
    } catch (error) {
      setStatusText(`Build preview failed: ${stringifyError(error)}`);
      return null;
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  async function fetchPreview(jobId: string): Promise<PreviewResponse | null> {
    try {
      const response = await fetch(`${apiBase}/jobs/${jobId}/preview`);
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      return (await response.json()) as PreviewResponse;
    } catch (error) {
      setStatusText(`Load preview failed: ${stringifyError(error)}`);
      return null;
    }
  }

  async function savePreviewEdits(jobId: string, edits: PreviewEdits): Promise<boolean> {
    setIsBusy(true);
    setBusyAction("Saving preview edits...");
    setStatusText("Saving preview edits...");
    try {
      const response = await fetch(`${apiBase}/jobs/${jobId}/preview-edits`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(edits),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      await refreshJob(jobId);
      setStatusText("Preview edits saved.");
      return true;
    } catch (error) {
      setStatusText(`Save preview edits failed: ${stringifyError(error)}`);
      return false;
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  async function convertFromIr(jobId: string): Promise<FromIrJobResponse | null> {
    setIsBusy(true);
    setBusyAction("Converting from SlideIR...");
    setStatusText(`Building PPTX from preview IR for job ${jobId}...`);
    try {
      const response = await fetch(`${apiBase}/jobs/${jobId}/from-ir`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as FromIrJobResponse;
      setConvertResult({
        job_id: payload.job_id,
        status: payload.status,
        output_pptx_path: payload.output_pptx_path,
        report_path: payload.report_path,
        page_count: payload.page_count,
      });
      await refreshJob(jobId);
      setStatusText(`Convert finished for ${payload.page_count} page(s) (from-ir).`);
      return payload;
    } catch (error) {
      setStatusText(`Convert from-ir failed: ${stringifyError(error)}`);
      return null;
    } finally {
      setIsBusy(false);
      setBusyAction(null);
    }
  }

  return {
    apiBase,
    busyAction,
    buildPreview,
    convertFromIr,
    convertResult,
    createJob,
    convertJob,
    fetchPreview,
    isBusy,
    job,
    refreshJob,
    runDetect,
    saveBoxes,
    savePreviewEdits,
    setJob,
    setStatusText,
    statusText,
  };
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as ApiErrorResponse;
    if (typeof payload.detail?.message === "string") {
      return payload.detail.message;
    }
  } catch {
    return response.text();
  }
  return response.statusText;
}

function stringifyError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}