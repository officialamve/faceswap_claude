"""Model file integrity and (optional) download from an administrator-controlled HTTPS URL.

No weights are bundled and no default download location is built in: model licences differ
from code licences (see README, "Model licensing")."""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import requests

from .config import ModelSpec

log = logging.getLogger(__name__)


class ModelError(Exception):
    pass


class ModelMissingError(ModelError):
    pass


class ModelIntegrityError(ModelError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(spec: ModelSpec, timeout: int = 600, max_bytes: int = 2_000_000_000) -> Path:
    """Return a verified local path to the model, downloading it first if a URL is configured."""
    if spec.path.exists():
        if sha256_file(spec.path) != spec.sha256:
            raise ModelIntegrityError(f"checksum mismatch for {spec.name} ({spec.path.name})")
        return spec.path
    if not spec.url:
        raise ModelMissingError(
            f"{spec.name} model file {spec.path.name} not found and no download URL configured")
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    part = spec.path.with_name(spec.path.name + ".part")
    log.info("downloading %s model (%s)", spec.name, spec.path.name)
    start = time.monotonic()
    h = hashlib.sha256()
    try:
        with requests.get(spec.url, stream=True, timeout=(10, 60)) as resp:
            resp.raise_for_status()
            total = 0
            with open(part, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    total += len(chunk)
                    if total > max_bytes or time.monotonic() - start > timeout:
                        raise ModelError(f"{spec.name} download exceeded size/time limit")
                    h.update(chunk)
                    fh.write(chunk)
    except requests.RequestException as exc:
        part.unlink(missing_ok=True)
        raise ModelError(f"{spec.name} download failed: {type(exc).__name__}") from None
    except ModelError:
        part.unlink(missing_ok=True)
        raise
    if h.hexdigest() != spec.sha256:
        part.unlink(missing_ok=True)
        raise ModelIntegrityError(f"checksum mismatch for downloaded {spec.name}")
    os.replace(part, spec.path)
    return spec.path
