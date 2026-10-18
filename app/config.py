"""Environment-driven configuration with fail-fast validation."""
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

MAX_PROCESSING_TIMEOUT = 600


class ConfigError(ValueError):
    """Raised when a required setting is missing or invalid."""


_TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")
_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{16,256}$")  # Telegram secret_token alphabet
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    sha256: str
    url: str | None = None


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    admin_id: int
    webhook_base_url: str
    webhook_secret: str = field(repr=False)
    port: int
    processing_timeout: int
    download_timeout: int
    max_input_bytes: int
    max_image_pixels: int
    max_image_dimension: int
    max_concurrent_jobs: int
    temp_dir: Path
    log_level: str
    models: tuple[ModelSpec, ...]

    @property
    def webhook_url(self) -> str:
        return self.webhook_base_url + "/telegram/webhook"

    def model(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}")
    return value


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required but not set")
    return value


def load_settings(env: Mapping[str, str]) -> Settings:
    """Parse settings from a mapping (normally os.environ). Never echoes secret values."""
    token = _required(env, "BOT_TOKEN")
    if not _TOKEN_RE.match(token):
        raise ConfigError("BOT_TOKEN does not look like a Telegram bot token")

    admin_raw = _required(env, "ADMIN_TELEGRAM_ID")
    if not re.fullmatch(r"[1-9]\d{0,15}", admin_raw):
        raise ConfigError("ADMIN_TELEGRAM_ID must be a positive numeric Telegram user ID")
    admin_id = int(admin_raw)

    secret = _required(env, "WEBHOOK_SECRET")
    if not _SECRET_RE.match(secret):
        raise ConfigError("WEBHOOK_SECRET must be 16-256 chars of A-Z a-z 0-9 _ -")

    base = (env.get("WEBHOOK_URL", "").strip() or env.get("RENDER_EXTERNAL_URL", "").strip()).rstrip("/")
    if not base:
        raise ConfigError("WEBHOOK_URL is required (or RENDER_EXTERNAL_URL on Render)")
    if not base.startswith("https://"):
        raise ConfigError("WEBHOOK_URL must start with https:// (Telegram requires HTTPS)")

    model_dir = Path(env.get("MODEL_DIR", "").strip() or "models")
    specs = []
    for name, default_file, prefix in (
        ("detector", "det_10g.onnx", "DET"),
        ("embedder", "w600k_r50.onnx", "REC"),
        ("swapper", "inswapper_128.onnx", "SWAP"),
    ):
        sha = _required(env, f"{prefix}_MODEL_SHA256").lower()
        if not _SHA_RE.match(sha):
            raise ConfigError(f"{prefix}_MODEL_SHA256 must be 64 hex characters")
        url = env.get(f"{prefix}_MODEL_URL", "").strip() or None
        if url and not url.startswith("https://"):
            raise ConfigError(f"{prefix}_MODEL_URL must use https://")
        file_name = env.get(f"{prefix}_MODEL_FILE", "").strip() or default_file
        if "/" in file_name or "\\" in file_name or file_name.startswith("."):
            raise ConfigError(f"{prefix}_MODEL_FILE must be a plain file name")
        specs.append(ModelSpec(name, model_dir / file_name, sha, url))

    temp = env.get("TEMP_DIR", "").strip()
    return Settings(
        bot_token=token,
        admin_id=admin_id,
        webhook_base_url=base,
        webhook_secret=secret,
        port=_int(env, "PORT", 10000, 1, 65535),
        processing_timeout=_int(env, "PROCESSING_TIMEOUT_SECONDS", 600, 1, MAX_PROCESSING_TIMEOUT),
        download_timeout=_int(env, "DOWNLOAD_TIMEOUT_SECONDS", 60, 1, 600),
        max_input_bytes=_int(env, "MAX_INPUT_BYTES", 20_000_000, 1_000, 20_000_000),
        max_image_pixels=_int(env, "MAX_IMAGE_PIXELS", 16_000_000, 10_000, 100_000_000),
        max_image_dimension=_int(env, "MAX_IMAGE_DIMENSION", 4096, 64, 16_384),
        max_concurrent_jobs=_int(env, "MAX_CONCURRENT_JOBS", 1, 1, 4),
        temp_dir=Path(temp) if temp else Path(tempfile.gettempdir()) / "faceswap-bot",
        log_level=(env.get("LOG_LEVEL", "").strip() or "INFO").upper(),
        models=tuple(specs),
    )
