import { afterEach, describe, expect, it, vi } from "vitest";

import {
  JOB_BUSY_RETRY_DELAYS_MS,
  JOB_BUSY_STATUS_MESSAGE,
  fetchWithJobBusyRetry,
  readApiErrorCode,
} from "./jobBusyRetry";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("jobBusyRetry", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("exposes the user-facing busy message", () => {
    expect(JOB_BUSY_STATUS_MESSAGE).toContain("busy with this job");
    expect(JOB_BUSY_RETRY_DELAYS_MS).toEqual([1000, 2000, 4000]);
  });

  it("reads detail.code from a cloned body", async () => {
    const response = jsonResponse(409, {
      detail: { code: "job-busy", message: "retry" },
    });
    await expect(readApiErrorCode(response)).resolves.toBe("job-busy");
    await expect(response.json()).resolves.toMatchObject({
      detail: { code: "job-busy" },
    });
  });

  it("retries job-busy until success", async () => {
    const fetchImpl = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(409, { detail: { code: "job-busy", message: "busy" } }),
      )
      .mockResolvedValueOnce(
        jsonResponse(409, { detail: { code: "job-busy", message: "busy" } }),
      )
      .mockResolvedValueOnce(jsonResponse(200, { ok: true }));
    const sleep = vi.fn().mockResolvedValue(undefined);
    const onBusyRetry = vi.fn();

    const response = await fetchWithJobBusyRetry("/jobs/x/preview", undefined, {
      fetchImpl,
      sleep,
      onBusyRetry,
      delaysMs: [10, 20, 40],
    });

    expect(response.status).toBe(200);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
    expect(sleep).toHaveBeenCalledTimes(2);
    expect(sleep).toHaveBeenNthCalledWith(1, 10);
    expect(sleep).toHaveBeenNthCalledWith(2, 20);
    expect(onBusyRetry).toHaveBeenCalledTimes(2);
    expect(onBusyRetry).toHaveBeenNthCalledWith(1, 1, 10);
  });

  it("exhausts retries and returns the final 409", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse(409, { detail: { code: "job-busy", message: "busy" } }),
    );
    const sleep = vi.fn().mockResolvedValue(undefined);

    const response = await fetchWithJobBusyRetry("/jobs/x/preview-edits", { method: "PUT" }, {
      fetchImpl,
      sleep,
      delaysMs: [1, 2, 3],
    });

    expect(response.status).toBe(409);
    expect(await readApiErrorCode(response)).toBe("job-busy");
    expect(fetchImpl).toHaveBeenCalledTimes(4);
    expect(sleep).toHaveBeenCalledTimes(3);
  });

  it("does not retry other errors", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse(400, { detail: { code: "input-error", message: "bad" } }),
    );
    const sleep = vi.fn().mockResolvedValue(undefined);

    const response = await fetchWithJobBusyRetry("/jobs/x/preview", undefined, {
      fetchImpl,
      sleep,
    });

    expect(response.status).toBe(400);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(sleep).not.toHaveBeenCalled();
  });

  it("does not retry when retryOnJobBusy is false", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse(409, { detail: { code: "job-busy", message: "busy" } }),
    );
    const sleep = vi.fn().mockResolvedValue(undefined);

    const response = await fetchWithJobBusyRetry("/jobs/x/from-ir", { method: "POST" }, {
      fetchImpl,
      sleep,
      retryOnJobBusy: false,
    });

    expect(response.status).toBe(409);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(sleep).not.toHaveBeenCalled();
  });

  it("does not retry a 409 with a different detail code", async () => {
    const fetchImpl = vi.fn().mockResolvedValue(
      jsonResponse(409, { detail: { code: "conflict", message: "other" } }),
    );
    const sleep = vi.fn().mockResolvedValue(undefined);

    const response = await fetchWithJobBusyRetry("/jobs/x/preview", undefined, {
      fetchImpl,
      sleep,
    });

    expect(response.status).toBe(409);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(sleep).not.toHaveBeenCalled();
  });
});
