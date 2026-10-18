"""HTTP server: /healthz, /readyz and the authenticated Telegram webhook."""
from __future__ import annotations

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

log = logging.getLogger(__name__)
MAX_BODY = 1_000_000


class WebhookApp(Protocol):
    def readiness(self) -> tuple[bool, str]: ...
    def submit_update(self, update: dict) -> bool: ...


def make_server(host: str, port: int, app: WebhookApp, secret: str) -> ThreadingHTTPServer:
    secret_b = secret.encode()

    class Handler(BaseHTTPRequestHandler):
        timeout = 30  # socket timeout vs slow clients
        server_version = "FaceSwapBot"
        sys_version = ""

        def log_message(self, fmt, *args):  # no request lines in logs
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path == "/healthz":
                self._send(200, {"status": "ok"})
            elif self.path == "/readyz":
                ready, reason = app.readiness()
                self._send(200 if ready else 503, {"ready": ready, "reason": reason})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/telegram/webhook":
                self._send(404, {"error": "not found"})
                return
            got = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "").encode()
            if not hmac.compare_digest(got, secret_b):
                self._send(403, {"error": "forbidden"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send(411, {"error": "length required"})
                return
            if length < 0 or length > MAX_BODY:
                self._send(413, {"error": "too large"})
                return
            try:
                update = json.loads(self.rfile.read(length))
            except (ValueError, OSError):
                self._send(400, {"error": "bad request"})
                return
            if not isinstance(update, dict):
                self._send(400, {"error": "bad request"})
                return
            # 200 only means "update accepted into the queue", never "swap succeeded".
            if app.submit_update(update):
                self._send(200, {"ok": True})
            else:
                self._send(503, {"error": "busy"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
