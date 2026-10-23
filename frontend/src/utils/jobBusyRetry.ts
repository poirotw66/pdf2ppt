/** Friendly status while auto-retrying a 409 job-busy response. */
export const JOB_BUSY_STATUS_MESSAGE =
  "The server is busy with this job, retrying shortly…";

/** Backoff delays (ms) between retries: 3 retries → 1s, 2s, 4s. */
export const JOB_BUSY_RETRY_DELAYS_MS = [1000, 2000, 4000] as const;

export type JobBusyRetryOptions = {
  /** When false, never retry (Convert / from-ir / upload). Default true. */
  retryOnJobBusy?: boolean;
  delaysMs?: readonly number[];
  sleep?: (ms: number) => Promise<void>;
  onBusyRetry?: (attempt: number, delayMs: number) => void;
  fetchImpl?: typeof fetch;
};

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export async function readApiErrorCode(response: Response): Promise<string | null> {
  try {
    const payload = (await response.clone().json()) as {
      detail?: { code?: string };
    };
    const code = payload.detail?.code;
    return typeof code === "string" ? code : null;
  } catch {
    return null;
  }
}

/**
 * fetch() wrapper that retries only on HTTP 409 with detail.code === "job-busy".
 * Use for idempotent GETs / PUT edits / preview build — not Convert or downloads.
 */
export async function fetchWithJobBusyRetry(
  input: RequestInfo | URL,
  init?: RequestInit,
  options: JobBusyRetryOptions = {},
): Promise<Response> {
  const {
    retryOnJobBusy = true,
    delaysMs = JOB_BUSY_RETRY_DELAYS_MS,
    sleep = defaultSleep,
    onBusyRetry,
    fetchImpl = fetch,
  } = options;

  const maxAttempts = retryOnJobBusy ? delaysMs.length + 1 : 1;
  let response = await fetchImpl(input, init);

  for (let attempt = 0; attempt < maxAttempts - 1; attempt += 1) {
    if (response.ok || response.status !== 409) {
      return response;
    }
    const code = await readApiErrorCode(response);
    if (code !== "job-busy") {
      return response;
    }
    const delayMs = delaysMs[attempt] ?? delaysMs[delaysMs.length - 1] ?? 1000;
    onBusyRetry?.(attempt + 1, delayMs);
    await sleep(delayMs);
    response = await fetchImpl(input, init);
  }

  return response;
}
