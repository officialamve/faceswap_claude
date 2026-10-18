"""Admin-only authorization. Identity is the numeric user ID, never a username."""
from __future__ import annotations

from typing import Any


def _int_id(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def extract_actor(update: dict) -> tuple[int | None, int | None, str | None]:
    """Return (user_id, chat_id, chat_type) for message / callback updates, else Nones."""
    if isinstance(update.get("message"), dict):
        msg, sender = update["message"], update["message"].get("from")
    elif isinstance(update.get("callback_query"), dict):
        cq = update["callback_query"]
        msg, sender = cq.get("message"), cq.get("from")
    else:
        return None, None, None
    if not isinstance(sender, dict) or not isinstance(msg, dict) or sender.get("is_bot"):
        return None, None, None
    chat = msg.get("chat")
    if not isinstance(chat, dict):
        return None, None, None
    return _int_id(sender.get("id")), _int_id(chat.get("id")), chat.get("type")


def is_authorized(update: dict, admin_id: int) -> bool:
    user_id, chat_id, chat_type = extract_actor(update)
    return user_id is not None and user_id == admin_id and chat_type == "private" and chat_id == admin_id
