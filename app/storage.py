"""Private temporary workspace with safe paths and cleanup."""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)
PREFIX = "sess-"
ALLOWED_NAMES = {"source.bin", "target.bin"}


class Workspace:
    def __init__(self, base: Path) -> None:
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.base = base.resolve()
        self._active: set[Path] = set()
        self._lock = threading.Lock()

    def create(self) -> Path:
        d = self.base / f"{PREFIX}{uuid.uuid4().hex}"
        d.mkdir(mode=0o700)
        with self._lock:
            self._active.add(d)
        return d

    def _check(self, d: Path) -> None:
        if d.parent != self.base or not d.name.startswith(PREFIX):
            raise ValueError("path is outside the managed workspace")

    def write_atomic(self, d: Path, name: str, data: bytes) -> Path:
        self._check(d)
        if name not in ALLOWED_NAMES:  # never derive file names from untrusted input
            raise ValueError("unexpected file name")
        tmp, final = d / f".{name}.tmp", d / name
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        return final

    def remove(self, d: Path | None) -> None:
        if d is None:
            return
        self._check(d)
        with self._lock:
            self._active.discard(d)
        try:
            shutil.rmtree(d)
        except FileNotFoundError:
            pass
        except OSError:
            log.exception("could not remove workspace directory")

    def cleanup_stale(self, max_age_seconds: float, now: float | None = None) -> int:
        """Delete abandoned session dirs that are not active and older than max_age."""
        now = time.time() if now is None else now
        removed = 0
        with self._lock:
            active = set(self._active)
        for entry in self.base.iterdir():
            if not entry.name.startswith(PREFIX) or entry in active or not entry.is_dir():
                continue
            try:
                if now - entry.stat().st_mtime >= max_age_seconds:
                    shutil.rmtree(entry)
                    removed += 1
            except OSError:
                log.exception("stale cleanup failed for one directory")
        return removed
