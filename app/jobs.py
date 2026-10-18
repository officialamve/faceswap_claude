"""Bounded job queue, worker threads, and hard per-job deadlines.

Durability: the queue is in memory only; queued/running jobs are lost on restart.
Native inference cannot be interrupted: on timeout/cancel the job is marked final immediately,
its late result is discarded, and the worker thread is freed only when the native call returns."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DELIVERING = "delivering"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class JobCancelled(Exception):
    pass


class DeadlineExceeded(Exception):
    pass


class DuplicateJobError(Exception):
    pass


class QueueFullError(Exception):
    pass


class ManagerClosedError(Exception):
    pass


@dataclass(eq=False)
class Job:
    id: str
    chat_id: int
    workdir: Path
    status: JobStatus = JobStatus.QUEUED
    deadline: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    timings: dict[str, float] = field(default_factory=dict)
    started: float | None = None

    def checkpoint(self, stage: str = "") -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise DeadlineExceeded(stage)
        if self.cancel_event.is_set():
            raise JobCancelled(stage)


Runner = Callable[[Job], Any]
Finished = Callable[[Job, JobStatus, Any], None]


class JobManager:
    def __init__(self, runner: Runner, on_finished: Finished, timeout_seconds: float, *,
                 on_started: Callable[[Job], None] | None = None, workers: int = 1, max_queued: int = 1):
        self._runner, self._on_finished, self._on_started = runner, on_finished, on_started
        self._timeout = timeout_seconds
        self._queue: queue.Queue[Job | None] = queue.Queue(maxsize=max_queued)
        self._lock = threading.Lock()
        self._active: dict[str, Job] = {}
        self._busy = 0
        self._closed = False
        self._threads = [threading.Thread(target=self._loop, name=f"job-worker-{i}", daemon=True)
                         for i in range(workers)]
        for t in self._threads:
            t.start()

    # ---------------------------------------------------------------- public API
    def submit(self, job: Job) -> None:
        with self._lock:
            if self._closed:
                raise ManagerClosedError()
            if job.id in self._active or any(j.chat_id == job.chat_id for j in self._active.values()):
                raise DuplicateJobError(job.id)
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                raise QueueFullError() from None
            self._active[job.id] = job

    def cancel(self, job_id: str | None) -> str:
        """Return 'queued' | 'running' | 'delivering' | 'none'."""
        with self._lock:
            job = self._active.get(job_id) if job_id else None
            if job is None:
                return "none"
            if job.status is JobStatus.DELIVERING:
                return "delivering"
            was = job.status
            job.status = JobStatus.CANCELLED
            job.cancel_event.set()
            self._active.pop(job.id, None)
        self._notify(job, JobStatus.CANCELLED, None)
        return "queued" if was is JobStatus.QUEUED else "running"

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"queued": self._queue.qsize(), "busy_workers": self._busy}

    def shutdown(self, grace: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            jobs = list(self._active.values())
            self._active.clear()
            live = []
            for j in jobs:
                if j.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    j.status = JobStatus.CANCELLED
                    j.cancel_event.set()
                    live.append(j)
        for j in live:
            self._notify(j, JobStatus.CANCELLED, "shutdown")
        for _ in self._threads:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        end = time.monotonic() + grace
        for t in self._threads:
            t.join(max(0.0, end - time.monotonic()))

    # ---------------------------------------------------------------- internals
    def _notify(self, job: Job, status: JobStatus, payload: Any) -> None:
        try:
            self._on_finished(job, status, payload)
        except Exception:
            log.exception("job finished-callback failed", extra={"job_id": job.id})

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            self._run(job)

    def _final(self, job: Job, status: JobStatus) -> bool:
        """Claim the job's final transition. True only for the first claimant."""
        with self._lock:
            if job.status is not JobStatus.RUNNING:
                return False
            job.status = JobStatus.DELIVERING if status is JobStatus.SUCCEEDED else status
            if status is not JobStatus.SUCCEEDED:
                self._active.pop(job.id, None)
            return True

    def _expire(self, job: Job) -> None:
        if self._final(job, JobStatus.TIMED_OUT):
            job.cancel_event.set()
            self._notify(job, JobStatus.TIMED_OUT, None)

    def _run(self, job: Job) -> None:
        with self._lock:
            if job.status is not JobStatus.QUEUED:  # cancelled while waiting
                return
            job.status = JobStatus.RUNNING
            job.started = time.monotonic()
            job.deadline = job.started + self._timeout
            self._busy += 1
        timer = threading.Timer(self._timeout, self._expire, args=(job,))
        timer.daemon = True
        timer.start()
        try:
            if self._on_started:
                try:
                    self._on_started(job)
                except Exception:
                    log.exception("on_started failed", extra={"job_id": job.id})
            try:
                result, error = self._runner(job), None
            except BaseException as exc:  # noqa: BLE001 - classified below, never swallowed silently
                result, error = None, exc
            timer.cancel()
            if error is None:
                if self._final(job, JobStatus.SUCCEEDED):
                    try:
                        self._notify(job, JobStatus.SUCCEEDED, result)
                    finally:
                        with self._lock:
                            job.status = JobStatus.SUCCEEDED
                            self._active.pop(job.id, None)
                else:
                    log.warning("discarding late result", extra={"job_id": job.id})
                return
            if isinstance(error, DeadlineExceeded):
                status = JobStatus.TIMED_OUT
            elif isinstance(error, JobCancelled):
                status = JobStatus.CANCELLED
            else:
                status = JobStatus.FAILED
            if self._final(job, status):
                self._notify(job, status, error)
            else:
                log.info("late %s for already-final job", type(error).__name__, extra={"job_id": job.id})
        finally:
            timer.cancel()
            with self._lock:
                self._busy -= 1
