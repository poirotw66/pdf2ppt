"""Bounded execution helpers for OCR / layout / subprocess steps.

Per-page OCR and layout can hang indefinitely on rare model / LibreOffice
pathologies. Callers wrap those steps with :func:`call_with_timeout` so a
stuck page is abandoned (logged) and the job continues with a graceful
fallback. Abandoned worker threads may linger until process exit; that is
preferable to hanging the whole conversion for tens of minutes.

LibreOffice invocations should use
:func:`pdf2ppt.render_compare.run_subprocess_with_process_group` so timeouts
reap the process tree.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TypeVar

logger = logging.getLogger(__name__)

DEFAULT_PAGE_TIMEOUT_SECONDS = 120.0

T = TypeVar("T")


def page_timeout_seconds_from_environ(*, default: float = DEFAULT_PAGE_TIMEOUT_SECONDS) -> float:
    """Read ``PDF2PPT_PAGE_TIMEOUT_S`` (seconds); ``<= 0`` disables the timeout."""

    raw = os.environ.get("PDF2PPT_PAGE_TIMEOUT_S")
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning("Invalid PDF2PPT_PAGE_TIMEOUT_S=%r; using default %.1f", raw, default)
        return float(default)


def call_with_timeout(
    func: Callable[..., T],
    timeout_s: float | None,
    /,
    *args: object,
    label: str = "operation",
    **kwargs: object,
) -> T:
    """Run ``func(*args, **kwargs)`` with an optional wall-clock timeout.

    When ``timeout_s`` is ``None`` or ``<= 0``, runs inline on the calling
    thread (byte-identical control flow for the happy path when disabled).
    When a positive timeout is set, runs in a worker thread and raises
    ``TimeoutError`` if it does not finish in time. The worker is not killed;
    callers must tolerate a lingering thread.
    """

    if timeout_s is None or float(timeout_s) <= 0:
        return func(*args, **kwargs)  # type: ignore[arg-type,call-arg,return-value]

    limit = float(timeout_s)
    # Do not use the executor as a context manager: on timeout its __exit__
    # would wait for the hung worker and re-block the job.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf2ppt-timeout")
    future = pool.submit(func, *args, **kwargs)  # type: ignore[arg-type]
    try:
        result = future.result(timeout=limit)
    except FuturesTimeoutError as error:
        future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"{label} timed out after {limit:.1f}s") from error
    except Exception:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=False)
        return result


__all__ = [
    "DEFAULT_PAGE_TIMEOUT_SECONDS",
    "call_with_timeout",
    "page_timeout_seconds_from_environ",
]
