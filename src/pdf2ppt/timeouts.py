"""Bounded execution helpers for OCR / layout / subprocess steps.

Per-page OCR and layout can hang indefinitely on rare model pathologies.
Callers wrap those steps with :func:`call_with_timeout` so a stuck page is
abandoned (logged) and the job continues with a graceful fallback.

Design constraints (verified against real PaddleOCR):

- Paddle predictors are **thread-affine**: every OCR ``predict`` must run on
  the same worker thread that first touched the engine. A fresh thread per
  call breaks multi-batch OCR (``std::exception`` on the second batch).
- Abandoned workers must be **daemon** threads so a timed-out busy-loop
  cannot keep the interpreter alive at 100% CPU after the job finishes.
- On timeout the lane is replaced; the abandoned daemon may linger, but the
  process can exit. Pair with the OCR predict lock so an abandoned wake-up
  cannot overlap the next ``predict``.

LibreOffice invocations should use
:func:`pdf2ppt.render_compare.run_subprocess_with_process_group` so timeouts
reap the process tree.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Callable
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


class _TimeoutLane:
    """Single persistent daemon worker; replaced only after a timeout."""

    def __init__(self, *, name: str = "pdf2ppt-timeout-lane") -> None:
        self._name = name
        self._submit_lock = threading.Lock()
        self._jobs: queue.Queue[object] | None = None
        self._worker: threading.Thread | None = None
        self._generation = 0

    def _spawn_locked(self) -> None:
        jobs: queue.Queue[object] = queue.Queue()
        self._jobs = jobs
        self._generation += 1
        generation = self._generation

        def run() -> None:
            while True:
                item = jobs.get()
                if item is None:
                    return
                assert isinstance(item, tuple)
                func_obj, call_args, call_kwargs, done_event, result_box = item
                assert callable(func_obj)
                assert isinstance(done_event, threading.Event)
                assert isinstance(result_box, dict)
                try:
                    result_box["result"] = func_obj(*call_args, **call_kwargs)
                    result_box["error"] = None
                except BaseException as error:  # noqa: BLE001 — deliver to caller
                    result_box["result"] = None
                    result_box["error"] = error
                finally:
                    done_event.set()

        worker = threading.Thread(
            target=run,
            name=f"{self._name}-{generation}",
            daemon=True,
        )
        worker.start()
        self._worker = worker

    def _ensure_worker_locked(self) -> queue.Queue[object]:
        if self._jobs is None or self._worker is None or not self._worker.is_alive():
            self._spawn_locked()
        assert self._jobs is not None
        return self._jobs

    def call(
        self,
        func: Callable[..., T],
        timeout_s: float,
        /,
        *args: object,
        label: str = "operation",
        **kwargs: object,
    ) -> T:
        done = threading.Event()
        box: dict[str, object] = {}
        with self._submit_lock:
            jobs = self._ensure_worker_locked()
            jobs.put((func, args, kwargs, done, box))

        if not done.wait(timeout=max(0.0, float(timeout_s))):
            # Abandon the hung daemon worker (it keeps the old queue). Spawn a
            # replacement lane so later calls stay thread-affine to a live worker.
            with self._submit_lock:
                self._spawn_locked()
            raise TimeoutError(f"{label} timed out after {float(timeout_s):.1f}s")

        error = box.get("error")
        if error is not None:
            assert isinstance(error, BaseException)
            raise error
        return box["result"]  # type: ignore[return-value]

    def reset_for_tests(self) -> None:
        """Drop the worker so the next call starts a fresh lane (unit tests)."""

        with self._submit_lock:
            if self._jobs is not None:
                try:
                    self._jobs.put_nowait(None)
                except Exception:  # noqa: BLE001
                    pass
            self._jobs = None
            self._worker = None


_DEFAULT_LANE = _TimeoutLane()


def call_with_timeout(
    func: Callable[..., T],
    timeout_s: float | None,
    /,
    *args: object,
    label: str = "operation",
    lane: _TimeoutLane | None = None,
    **kwargs: object,
) -> T:
    """Run ``func(*args, **kwargs)`` with an optional wall-clock timeout.

    When ``timeout_s`` is ``None`` or ``<= 0``, runs inline on the calling
    thread (byte-identical control flow when the timeout is disabled).

    When a positive timeout is set, runs on a **persistent daemon worker
    lane** so thread-affine native engines (PaddleOCR) keep working across
    batches. On timeout the lane is replaced; the abandoned worker is daemon
    so it cannot block interpreter exit.
    """

    if timeout_s is None or float(timeout_s) <= 0:
        return func(*args, **kwargs)  # type: ignore[arg-type,call-arg,return-value]

    active = lane if lane is not None else _DEFAULT_LANE
    return active.call(func, float(timeout_s), *args, label=label, **kwargs)


def reset_timeout_lane_for_tests() -> None:
    """Reset the process-wide timeout lane (tests only)."""

    _DEFAULT_LANE.reset_for_tests()


__all__ = [
    "DEFAULT_PAGE_TIMEOUT_SECONDS",
    "call_with_timeout",
    "page_timeout_seconds_from_environ",
    "reset_timeout_lane_for_tests",
]
