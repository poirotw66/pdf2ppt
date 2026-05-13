import { useState } from "react";

import { detectConfidenceThreshold } from "../config";
import type { ConvertResponse, DetectResponse, JobResponse, PagePayload } from "../types";

const apiBase = import.meta.env.VITE_API_BASE_URL ?? "";

export function usePdf2PptApi() {
  const [job, setJob] = useState<JobResponse | null>(null);
  const [statusText, setStatusText] = useState("Upload a PDF to start.");
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

  async function convertJob(jobId: string): Promise<ConvertResponse | null> {
    setIsBusy(true);
    setBusyAction("Converting PPTX...");
    setStatusText(`Converting job ${jobId} into PPTX...`);
    try {
      const response = await fetch(`${apiBase}/jobs/${jobId}/convert`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ write_debug_artifacts: false }),
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

  return {
    apiBase,
    busyAction,
    convertResult,
    createJob,
    convertJob,
    isBusy,
    job,
    refreshJob,
    runDetect,
    saveBoxes,
    setJob,
    setStatusText,
    statusText,
  };
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string | { message?: string } };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (payload.detail && typeof payload.detail.message === "string") {
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