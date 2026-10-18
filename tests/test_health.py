import http.client
import json
import threading
import unittest

from app.health import make_server
from tests.helpers import SECRET


class FakeApp:
    def __init__(self):
        self.ready, self.updates, self.full = False, [], False

    def readiness(self):
        return self.ready, "ready" if self.ready else "loading_models"

    def submit_update(self, u):
        if self.full:
            return False
        self.updates.append(u)
        return True


class HealthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FakeApp()
        cls.server = make_server("127.0.0.1", 0, cls.app, SECRET)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def post(self, body, secret=SECRET):
        h = {"Content-Type": "application/json"}
        if secret is not None:
            h["X-Telegram-Bot-Api-Secret-Token"] = secret
        return self.req("POST", "/telegram/webhook", body, h)

    def test_healthz_and_readyz(self):
        self.assertEqual(self.req("GET", "/healthz")[0], 200)
        self.app.ready = False
        code, data = self.req("GET", "/readyz")
        self.assertEqual(code, 503)
        self.assertFalse(json.loads(data)["ready"])
        self.app.ready = True
        self.assertEqual(self.req("GET", "/readyz")[0], 200)
        self.app.ready = False

    def test_webhook_secret_required(self):
        before = len(self.app.updates)
        self.assertEqual(self.post("{}", secret=None)[0], 403)
        self.assertEqual(self.post("{}", secret="wrong")[0], 403)
        self.assertEqual(len(self.app.updates), before)

    def test_webhook_accepts_valid(self):
        self.assertEqual(self.post('{"update_id": 5}')[0], 200)
        self.assertEqual(self.app.updates[-1], {"update_id": 5})

    def test_webhook_bad_bodies(self):
        self.assertEqual(self.post("not json")[0], 400)
        self.assertEqual(self.post("[1,2]")[0], 400)

    def test_webhook_oversize_and_busy(self):
        self.assertEqual(self.post("x" * 1_100_000)[0], 413)
        self.app.full = True
        self.assertEqual(self.post("{}")[0], 503)
        self.app.full = False

    def test_unknown_paths(self):
        self.assertEqual(self.req("GET", "/telegram/webhook")[0], 404)
        self.assertEqual(self.req("GET", "/swap")[0], 404)
        self.assertEqual(self.req("POST", "/other", "{}")[0], 404)


if __name__ == "__main__":
    unittest.main()
