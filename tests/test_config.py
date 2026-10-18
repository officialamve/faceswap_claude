import unittest

from app.config import ConfigError, load_settings
from tests.helpers import SECRET, TOKEN, make_env


class ConfigTests(unittest.TestCase):
    def test_valid(self):
        s = load_settings(make_env())
        self.assertEqual(s.admin_id, 4242)
        self.assertEqual(s.processing_timeout, 600)
        self.assertEqual(s.webhook_url, "https://example.onrender.com/telegram/webhook")
        self.assertEqual(s.max_concurrent_jobs, 1)

    def test_repr_hides_secrets(self):
        r = repr(load_settings(make_env()))
        self.assertNotIn(TOKEN, r)
        self.assertNotIn(SECRET, r)

    def test_missing_required(self):
        for key in ("BOT_TOKEN", "ADMIN_TELEGRAM_ID", "WEBHOOK_SECRET", "SWAP_MODEL_SHA256"):
            env = make_env()
            del env[key]
            with self.assertRaises(ConfigError, msg=key):
                load_settings(env)

    def test_malformed_admin_id(self):
        for bad in ("abc", "0", "-5", "1.5", "12 3", "@name", "٣٤"):
            with self.assertRaises(ConfigError, msg=bad):
                load_settings(make_env(ADMIN_TELEGRAM_ID=bad))

    def test_bad_token_and_secret(self):
        with self.assertRaises(ConfigError):
            load_settings(make_env(BOT_TOKEN="nope"))
        with self.assertRaises(ConfigError):
            load_settings(make_env(WEBHOOK_SECRET="has space and !!"))
        with self.assertRaises(ConfigError):
            load_settings(make_env(WEBHOOK_SECRET="short"))

    def test_timeout_bounds(self):
        with self.assertRaises(ConfigError):
            load_settings(make_env(PROCESSING_TIMEOUT_SECONDS="601"))
        with self.assertRaises(ConfigError):
            load_settings(make_env(PROCESSING_TIMEOUT_SECONDS="0"))
        self.assertEqual(load_settings(make_env(PROCESSING_TIMEOUT_SECONDS="120")).processing_timeout, 120)

    def test_webhook_url_rules(self):
        with self.assertRaises(ConfigError):
            load_settings(make_env(WEBHOOK_URL="http://insecure.example"))
        env = make_env()
        del env["WEBHOOK_URL"]
        with self.assertRaises(ConfigError):
            load_settings(env)
        env["RENDER_EXTERNAL_URL"] = "https://x.onrender.com/"
        self.assertEqual(load_settings(env).webhook_base_url, "https://x.onrender.com")

    def test_model_settings(self):
        with self.assertRaises(ConfigError):
            load_settings(make_env(SWAP_MODEL_SHA256="xyz"))
        with self.assertRaises(ConfigError):
            load_settings(make_env(SWAP_MODEL_FILE="../evil.onnx"))
        with self.assertRaises(ConfigError):
            load_settings(make_env(SWAP_MODEL_URL="http://plain.example/m.onnx"))

    def test_limits_validated(self):
        with self.assertRaises(ConfigError):
            load_settings(make_env(MAX_INPUT_BYTES="999999999"))
        with self.assertRaises(ConfigError):
            load_settings(make_env(PORT="abc"))


if __name__ == "__main__":
    unittest.main()
