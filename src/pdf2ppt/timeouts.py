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
- The timeout clock starts when the worker **begins** the call (not at
  enqueue), so concurrent jobs sharing the lane do not time each other out.
- Queued jobs live in one shared deque and move to the replacement worker;
  an abandoned worker retires once its hung call returns (generation check)
  and drops references to call arguments so page images are not pinned.
- Nested ``call_with_timeout`` from the worker thread runs inline.

Known limitation (PaddleOCR): after a timeout the lane gets a new worker
thread. Predictors are thread-affine; ``predict`` on an engine built on a
previous thread can raise ``std::exception`` on some replacements (~30–40%).
Pipeline callers catch that as ``OcrProcessingError`` whenever the lane has
ever been replaced (``timeout_lane_generation() > 1``), including later jobs
that did not cause the swap, and degrade to empty OCR rather than aborting
with HTTP 502. Prefer keeping ``PDF2PPT_PAGE_TIMEOUT_S`` generous (default
120) so timeouts (and thus thread swaps) stay rare.

LibreOffice invocations should use
:func:`pdf2ppt.render_compare.run_subprocess_with_process_group` so timeouts
reap the process tree.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

DEFAULT_PAGE_TIMEOUT_SECONDS = 120.0
# Engine init (model load) is excluded from the first timed OCR call.
DEFAULT_OCR_INIT_TIMEOUT_SECONDS = 300.0

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


def ocr_init_timeout_seconds(page_timeout_s: float | None) -> float | None:
    """Timeout for a one-shot OCR engine warm-up on the shared lane.

    Returns ``None`` / ``<= 0`` semantics matching ``call_with_timeout`` when
    page timeouts are disabled; otherwise ``max(page_timeout, 300)`` so model
    load is not charged against the first batch.
    """

    if page_timeout_s is None or float(page_timeout_s) <= 0:
        return page_timeout_s
    return max(float(page_timeout_s), float(DEFAULT_OCR_INIT_TIMEOUT_SECONDS))


class _Job:
    """One submitted call. ``started`` is set when a worker actually begins it."""

    __slots__ = (
        "func",
        "args",
        "kwargs",
        "started",
        "done",
        "result",
        "error",
        "running",
        "abandoned",
    )

    def __init__(
        self,
        func: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        self.func: Callable[..., object] | None = func
        self.args: tuple[object, ...] | None = args
        self.kwargs: dict[str, object] | None = kwargs
        self.started = threading.Event()
        self.done = threading.Event()
        self.result: object = None
        self.error: BaseException | None = None
        self.running = False
        self.abandoned = False


class _TimeoutLane:
    """Single persistent daemon worker; replaced only after a timeout.

    * The timeout clock starts when the worker *begins* the call, not when it is
      queued, so concurrent API jobs sharing the lane do not time each other out.
    * Jobs still queued when a worker is abandoned are handed to the replacement
      worker (they live in one shared deque, not in a per-worker queue).
    * An abandoned worker exits as soon as its hung call finally returns (it sees
      that it is no longer the current generation); it drops references to the
      call arguments/result so a hung call does not pin page images forever.
    * A call made *from* the worker thread (nesting) runs inline instead of
      deadlocking on its own lane.

    Known limitation: after a timeout the lane gets a **new** worker thread.
    PaddleOCR predictors are thread-affine; calling an engine constructed on a
    previous thread can raise ``std::exception`` on some replacements. Callers
    should catch ``OcrProcessingError`` once ``generation() > 1`` (lane has
    ever been replaced) and degrade (empty OCR) rather than abort the whole
    job — including subsequent jobs that share the process-wide lane.
    """

    def __init__(self, *, name: str = "pdf2ppt-timeout-lane") -> None:
        self._name = name
        self._cv = threading.Condition(threading.Lock())
        self._pending: collections.deque[_Job] = collections.deque()
        self._worker: threading.Thread | None = None
        self._generation = 0
        self._abandoned_running = 0  # timed-out calls whose worker has not returned yet

    def abandoned_running(self) -> int:
        with self._cv:
            return self._abandoned_running

    def generation(self) -> int:
        with self._cv:
            return self._generation

    def _spawn_locked(self) -> None:
        self._generation += 1
        generation = self._generation

        def run() -> None:
            while True:
                with self._cv:
                    while not self._pending and generation == self._generation:
                        self._cv.wait()
                    if generation != self._generation:
                        return  # replaced after a timeout: retire
                    job = self._pending.popleft()
                    job.running = True
                job.started.set()
                func, args, kwargs = job.func, job.args, job.kwargs
                job.func = job.args = job.kwargs = None
                try:
                    assert func is not None and args is not None and kwargs is not None
                    job.result = func(*args, **kwargs)
                except BaseException as error:  # noqa: BLE001 — deliver to caller
                    job.error = error
                finally:
                    del func, args, kwargs
                    with self._cv:  # done+abandoned must change atomically w.r.t. the caller's check
                        if job.abandoned:
                            # Clamp: reset_for_tests() may zero the counter while abandoned
                            # workers are still finishing, which must not go negative.
                            self._abandoned_running = max(0, self._abandoned_running - 1)
                        job.done.set()

        worker = threading.Thread(target=run, name=f"{self._name}-{generation}", daemon=True)
        self._worker = worker
        worker.start()

    def call(
        self,
        func: Callable[..., T],
        timeout_s: float,
        /,
        *args: object,
        label: str = "operation",
        **kwargs: object,
    ) -> T:
        worker = self._worker
        if worker is not None and threading.current_thread() is worker:
            return func(*args, **kwargs)  # nested call from the lane itself

        job = _Job(func, args, kwargs)
        with self._cv:
            self._pending.append(job)
            if self._worker is None or not self._worker.is_alive():
                self._spawn_locked()
            self._cv.notify_all()

        # Queue wait is bounded by the timeout of whichever call is running ahead
        # of us (its caller replaces the worker, which then picks us up).
        job.started.wait()
        if not job.done.wait(timeout=max(0.0, float(timeout_s))):
            with self._cv:
                if not job.done.is_set():
                    # Abandon the hung worker; it retires when its call returns.
                    job.abandoned = True
                    self._abandoned_running += 1
                    self._spawn_locked()
                    self._cv.notify_all()
                    raise TimeoutError(f"{label} timed out after {float(timeout_s):.1f}s")

        error = job.error
        result = job.result
        job.result = job.error = None
        if error is not None:
            raise error
        return result  # type: ignore[return-value]

    def reset_for_tests(self) -> None:
        """Drop the worker so the next call starts a fresh lane (unit tests)."""

        with self._cv:
            self._generation += 1  # retires the current worker
            self._pending.clear()
            self._worker = None
            self._abandoned_running = 0
            self._cv.notify_all()


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
    batches. The clock starts when the worker begins the call. On timeout the
    lane is replaced; the abandoned worker is daemon and retires when its
    hung call finally returns.
    """

    if timeout_s is None or float(timeout_s) <= 0:
        return func(*args, **kwargs)  # type: ignore[arg-type,call-arg,return-value]

    active = lane if lane is not None else _DEFAULT_LANE
    return active.call(func, float(timeout_s), *args, label=label, **kwargs)


def abandoned_calls_running() -> int:
    """Number of timed-out calls whose abandoned worker is still inside the call."""

    return _DEFAULT_LANE.abandoned_running()


def timeout_lane_generation() -> int:
    """Monotonic generation of the process-wide timeout lane (increments on spawn)."""

    return _DEFAULT_LANE.generation()


def reset_timeout_lane_for_tests() -> None:
    """Reset the process-wide timeout lane (tests only)."""

    _DEFAULT_LANE.reset_for_tests()


__all__ = [
    "DEFAULT_OCR_INIT_TIMEOUT_SECONDS",
    "DEFAULT_PAGE_TIMEOUT_SECONDS",
    "abandoned_calls_running",
    "call_with_timeout",
    "ocr_init_timeout_seconds",
    "page_timeout_seconds_from_environ",
    "reset_timeout_lane_for_tests",
    "timeout_lane_generation",
]
