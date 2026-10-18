"""Shared test fakes. Everything here is MOCKED (Telegram and the inference engine)."""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

from app.config import load_settings
from app.face_swap import SwapOutput
from app.telegram_api import TelegramError

ADMIN = 4242
TOKEN = "123456:" + "A" * 35
SECRET = "s" * 24
SHA = "0" * 64


def make_env(**overrides) -> dict:
    env = {"BOT_TOKEN": TOKEN, "ADMIN_TELEGRAM_ID": str(ADMIN), "WEBHOOK_URL": "https://example.onrender.com",
           "WEBHOOK_SECRET": SECRET, "DET_MODEL_SHA256": SHA, "REC_MODEL_SHA256": SHA,
           "SWAP_MODEL_SHA256": SHA}
    env.update(overrides)
    return env


def make_settings(tmp_path, **env_overrides):
    return load_settings(make_env(TEMP_DIR=str(tmp_path), **env_overrides))


def image_bytes(fmt="JPEG", size=(256, 192), seed=0, exif_orientation=None) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class FakeTelegram:
    def __init__(self):
        self.messages, self.images, self.downloads, self.callbacks = [], [], [], []
        self.files: dict[str, bytes] = {}
        self.fail_upload = False

    def send_message(self, chat_id, text, keyboard=None):
        self.messages.append((chat_id, text))

    def answer_callback(self, cid, text=""):
        self.callbacks.append((cid, text))

    def download_file(self, file_id, max_bytes):
        self.downloads.append(file_id)
        return self.files[file_id]

    def send_image(self, chat_id, data, filename, mime, caption, as_photo):
        if self.fail_upload:
            raise TelegramError("sendPhoto: boom")
        self.images.append((chat_id, data, caption, as_photo))

    def texts(self):
        return " | ".join(t for _, t in self.messages)


class FakeSwapper:
    """MOCKED engine: returns the target with a block painted so output differs from input."""

    def __init__(self, error=None, delay=0.0, gate=None):
        self.error, self.delay, self.gate = error, delay, gate

    def swap(self, source, target, checkpoint):
        checkpoint("start")
        if self.gate is not None:
            self.gate.wait(10)
        if self.delay:
            time.sleep(self.delay)  # simulates a non-interruptible native call
        if self.error:
            raise self.error
        out = target.copy()
        out[10:40, 10:40] = 255 - out[10:40, 10:40]
        return SwapOutput(out, {"inference": 0.0})


def msg_update(update_id, user_id=ADMIN, text=None, photo_id=None, chat_type="private"):
    msg = {"message_id": update_id, "from": {"id": user_id, "is_bot": False},
           "chat": {"id": user_id, "type": chat_type}}
    if text is not None:
        msg["text"] = text
    if photo_id:
        msg["photo"] = [{"file_id": photo_id, "file_size": 1000, "width": 256, "height": 192}]
    return {"update_id": update_id, "message": msg}


def cb_update(update_id, data, user_id=ADMIN):
    return {"update_id": update_id, "callback_query": {
        "id": f"cb{update_id}", "from": {"id": user_id, "is_bot": False}, "data": data,
        "message": {"message_id": 1, "chat": {"id": user_id, "type": "private"}}}}
