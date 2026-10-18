"""Minimal Telegram Bot API client (requests). Never leaks the token in errors or logs."""
from __future__ import annotations

import time

import requests

API = "https://api.telegram.org"


class TelegramError(Exception):
    """Sanitised error: never contains the request URL (which embeds the bot token)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class FileTooLarge(TelegramError):
    pass


class TelegramClient:
    def __init__(self, token: str, download_timeout: int = 60, session: requests.Session | None = None):
        self._base = f"{API}/bot{token}"
        self._file_base = f"{API}/file/bot{token}"
        self._download_timeout = download_timeout
        self._http = session or requests.Session()

    def _call(self, method: str, *, payload: dict | None = None, files: dict | None = None,
              timeout: float = 30):
        try:
            if files:
                resp = self._http.post(f"{self._base}/{method}", data=payload, files=files,
                                       timeout=(10, max(timeout, 120)))
            else:
                resp = self._http.post(f"{self._base}/{method}", json=payload, timeout=(10, timeout))
        except requests.RequestException as exc:
            raise TelegramError(f"{method}: network error ({type(exc).__name__})") from None
        try:
            body = resp.json()
        except ValueError:
            raise TelegramError(f"{method}: invalid response", resp.status_code) from None
        if not body.get("ok"):
            raise TelegramError(f"{method}: {body.get('description', 'error')}", resp.status_code)
        return body.get("result")

    def get_me(self) -> dict:
        return self._call("getMe")

    def set_webhook(self, url: str, secret: str) -> None:
        self._call("setWebhook", payload={"url": url, "secret_token": secret,
                                          "allowed_updates": ["message", "callback_query"],
                                          "drop_pending_updates": False, "max_connections": 2})

    def send_message(self, chat_id: int, text: str, keyboard: dict | None = None) -> None:
        payload = {"chat_id": chat_id, "text": text}
        if keyboard:
            payload["reply_markup"] = keyboard
        self._call("sendMessage", payload=payload)

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        self._call("answerCallbackQuery", payload={"callback_query_id": callback_id, "text": text})

    def send_image(self, chat_id: int, data: bytes, filename: str, mime: str, caption: str,
                   as_photo: bool) -> None:
        field, method = ("photo", "sendPhoto") if as_photo else ("document", "sendDocument")
        self._call(method, payload={"chat_id": str(chat_id), "caption": caption},
                   files={field: (filename, data, mime)})

    def download_file(self, file_id: str, max_bytes: int) -> bytes:
        info = self._call("getFile", payload={"file_id": file_id})
        path = (info or {}).get("file_path")
        if not path or path.startswith("/") or ".." in path:
            raise TelegramError("file reference is unavailable or expired")
        if (info.get("file_size") or 0) > max_bytes:
            raise FileTooLarge("file exceeds the size limit")
        deadline = time.monotonic() + self._download_timeout
        try:
            with self._http.get(f"{self._file_base}/{path}", stream=True,
                                timeout=(10, min(30, self._download_timeout))) as resp:
                if resp.status_code != 200:
                    raise TelegramError("download failed", resp.status_code)
                chunks, total = [], 0
                for chunk in resp.iter_content(64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise FileTooLarge("file exceeds the size limit")
                    if time.monotonic() > deadline:
                        raise TelegramError("download timed out")
                    chunks.append(chunk)
        except requests.RequestException as exc:
            raise TelegramError(f"download: network error ({type(exc).__name__})") from None
        return b"".join(chunks)


def keyboard_json(rows: list[list[tuple[str, str]]]) -> dict:
    return {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in rows]}

