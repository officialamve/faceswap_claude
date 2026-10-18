"""Entry point: python -m app.main"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading

from .config import ConfigError, Settings, load_settings
from .face_swap import ModelInitError, load_swapper
from .handlers import BotApp
from .health import make_server
from .logging_config import configure_logging
from .models import ModelError
from .storage import Workspace
from .telegram_api import TelegramClient, TelegramError

log = logging.getLogger("app.main")
CLEANUP_INTERVAL = 600


def initialize(app: BotApp, settings: Settings, tg: TelegramClient) -> None:
    """Runs in a background thread so the port is bound (and /healthz answers) immediately."""
    try:
        tg.get_me()  # validates BOT_TOKEN
        app.swapper = load_swapper(settings)
        tg.set_webhook(settings.webhook_url, settings.webhook_secret)
    except TelegramError as exc:
        app.init_error = f"telegram: {exc}"
    except (ModelError, ModelInitError) as exc:
        app.init_error = f"model: {exc}"
    except Exception:
        log.exception("unexpected initialization failure")
        app.init_error = "unexpected error"
    if app.init_error:
        log.error("initialization failed: %s (service stays up but NOT ready; no automatic retry)",
                  app.init_error)
        try:
            tg.send_message(settings.admin_id, "Bot failed to initialize; see server logs. "
                                               "It will not accept swaps until fixed and redeployed.")
        except TelegramError:
            pass
        return
    app.mark_online()
    log.info("ready")
    try:
        tg.send_message(settings.admin_id, "Bot is online. If a job was running before a restart, it was lost.")
    except TelegramError:
        pass


def main() -> int:
    try:
        settings = load_settings(os.environ)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level, [settings.bot_token, settings.webhook_secret])
    ws = Workspace(settings.temp_dir)
    log.info("removed %d abandoned workspace dirs at startup", ws.cleanup_stale(0))
    tg = TelegramClient(settings.bot_token, settings.download_timeout)
    app = BotApp(settings, tg, ws)
    app.start()
    server = make_server("0.0.0.0", settings.port, app, settings.webhook_secret)
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    log.info("listening on port %d", settings.port)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    threading.Thread(target=initialize, args=(app, settings, tg), name="init", daemon=True).start()
    while not stop.wait(CLEANUP_INTERVAL):
        n = ws.cleanup_stale(settings.processing_timeout + CLEANUP_INTERVAL)
        if n:
            log.info("periodic cleanup removed %d dirs", n)
    log.info("shutting down")
    app.stop()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
