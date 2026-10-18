"""Structured key=value logging with secret redaction."""
from __future__ import annotations

import logging
import re

_REDACT = re.compile(r"bot\d{5,}:[A-Za-z0-9_-]{20,}|\d{5,}:[A-Za-z0-9_-]{30,}")


class RedactFilter(logging.Filter):
    def __init__(self, secrets: list[str] | None = None):
        super().__init__()
        self._secrets = [s for s in (secrets or []) if s]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = _REDACT.sub("[REDACTED]", record.getMessage())
        for s in self._secrets:
            msg = msg.replace(s, "[REDACTED]")
        record.msg, record.args = msg, ()
        return True


class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        job = getattr(record, "job_id", "-")
        base = (f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')}Z level={record.levelname} "
                f"logger={record.name} job={job} msg={record.getMessage()!r}")
        if record.exc_info:
            base += " exc=" + self.formatException(record.exc_info).replace("\n", " | ")
        return base


def configure_logging(level: str = "INFO", secrets: list[str] | None = None) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(KVFormatter())
    handler.addFilter(RedactFilter(secrets))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # its DEBUG lines contain request URLs
