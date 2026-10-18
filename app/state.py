"""Explicit conversation state machine and duplicate-update filter."""
from __future__ import annotations

import threading
from collections import OrderedDict
from enum import Enum
from pathlib import Path


class State(str, Enum):
    IDLE = "idle"
    AWAIT_SOURCE = "awaiting source photo"
    AWAIT_TARGET = "awaiting target photo"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


_ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.AWAIT_SOURCE},
    State.AWAIT_SOURCE: {State.AWAIT_TARGET, State.CANCELLED, State.FAILED},
    State.AWAIT_TARGET: {State.QUEUED, State.CANCELLED, State.FAILED},
    State.QUEUED: {State.PROCESSING, State.CANCELLED, State.FAILED},
    State.PROCESSING: {State.COMPLETED, State.CANCELLED, State.FAILED},
    State.COMPLETED: {State.AWAIT_SOURCE},
    State.CANCELLED: {State.AWAIT_SOURCE},
    State.FAILED: {State.AWAIT_SOURCE},
}


class InvalidTransition(Exception):
    pass


class Session:
    """Thread-safe state for the single administrator."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.state = State.IDLE
        self.workdir: Path | None = None
        self.job_id: str | None = None

    def transition(self, new: State, *, workdir: Path | None = None, job_id: str | None = None,
                   only_if_job: str | None = None) -> bool:
        """Move to `new`. With only_if_job, skip (return False) if the session now belongs
        to a different job, e.g. after /reset."""
        with self._lock:
            if only_if_job is not None and self.job_id != only_if_job:
                return False
            if new not in _ALLOWED[self.state]:
                raise InvalidTransition(f"{self.state.name} -> {new.name}")
            self.state = new
            if workdir is not None:
                self.workdir = workdir
            if job_id is not None:
                self.job_id = job_id
            return True

    def reset(self) -> tuple[Path | None, str | None]:
        """Unconditional return to IDLE; returns the workdir / job id that were active."""
        with self._lock:
            old = (self.workdir, self.job_id)
            self.state, self.workdir, self.job_id = State.IDLE, None, None
            return old

    def snapshot(self) -> tuple[State, Path | None, str | None]:
        with self._lock:
            return self.state, self.workdir, self.job_id

    @property
    def busy(self) -> bool:
        return self.state in (State.QUEUED, State.PROCESSING)


class UpdateDeduper:
    """Bounded memory of recently seen Telegram update_ids (idempotent handling)."""

    def __init__(self, capacity: int = 2048) -> None:
        self._seen: OrderedDict[int, None] = OrderedDict()
        self._cap = capacity
        self._lock = threading.Lock()

    def seen(self, update_id: int) -> bool:
        with self._lock:
            if update_id in self._seen:
                return True
            self._seen[update_id] = None
            if len(self._seen) > self._cap:
                self._seen.popitem(last=False)
            return False
