"""End-to-end bot flow with MOCKED Telegram and a MOCKED inference engine."""
import dataclasses
import tempfile
import unittest
from pathlib import Path

from app.face_swap import AmbiguousFaceError, NoFaceError
from app.handlers import BotApp
from app.state import State
from app.storage import Workspace
from tests.helpers import (ADMIN, FakeSwapper, FakeTelegram, cb_update, image_bytes, make_settings,
                           msg_update, wait_for)


class Base(unittest.TestCase):
    swapper = None
    timeout = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        s = make_settings(self._tmp.name)
        if self.timeout:
            s = dataclasses.replace(s, processing_timeout=self.timeout)
        self.tg = FakeTelegram()
        self.tg.files = {"src": image_bytes(seed=1), "tgt": image_bytes(seed=2), "bad": b"junk" * 50}
        self.ws = Workspace(Path(self._tmp.name) / "ws")
        self.app = BotApp(s, self.tg, self.ws)
        self.app.swapper = self.swapper or FakeSwapper()
        self.app.mark_online()
        self.addCleanup(self.app.jobs.shutdown, 0.2)
        self.n = 0

    def send(self, **kw):
        self.n += 1
        self.app.handle_update(msg_update(self.n, **kw))

    def press(self, data):
        self.n += 1
        self.app.handle_update(cb_update(self.n, data))

    def begin(self):
        self.press("swap")
        self.send(photo_id="src")
        self.send(photo_id="tgt")

    def leftovers(self):
        return [p for p in self.ws.base.iterdir()]


class FlowTests(Base):
    def test_success_and_cleanup(self):
        self.begin()
        self.assertTrue(wait_for(lambda: self.tg.images))
        self.assertTrue(wait_for(lambda: self.app.session.state is State.COMPLETED))
        _, data, caption, as_photo = self.tg.images[0]
        self.assertTrue(data.startswith(b"\xff\xd8"))
        self.assertIn("completed", caption)
        self.assertEqual(self.leftovers(), [])
        self.assertIn("Processing image", self.tg.texts())

    def test_unauthorized_rejected_before_download(self):
        self.app.handle_update(msg_update(1, user_id=999, photo_id="src"))
        self.app.handle_update(cb_update(2, "cancel", user_id=999))
        self.app.handle_update(msg_update(3, user_id=999, text="/reset"))
        self.assertEqual(self.tg.downloads, [])
        self.assertEqual(self.app.session.state, State.IDLE)
        self.assertNotIn(str(ADMIN), self.tg.texts())

    def test_photo_before_start_is_ignored(self):
        self.send(photo_id="src")
        self.assertEqual(self.tg.downloads, [])
        self.assertIn("Start Face Swap", self.tg.texts())

    def test_invalid_source_rejected_and_state_kept(self):
        self.press("swap")
        self.send(photo_id="bad")
        self.assertEqual(self.app.session.state, State.AWAIT_SOURCE)
        self.assertIn("rejected", self.tg.texts())
        self.send(photo_id="src")
        self.assertEqual(self.app.session.state, State.AWAIT_TARGET)

    def test_invalid_target_rejected(self):
        self.press("swap")
        self.send(photo_id="src")
        self.send(photo_id="bad")
        self.assertEqual(self.app.session.state, State.AWAIT_TARGET)
        self.assertIn("target photograph was rejected", self.tg.texts())

    def test_duplicate_update_ignored(self):
        u = cb_update(77, "swap")
        self.app.handle_update(u)
        n = len(self.tg.messages)
        self.app.handle_update(u)
        self.assertEqual(len(self.tg.messages), n)

    def test_reset_removes_files(self):
        self.press("swap")
        self.send(photo_id="src")
        self.assertEqual(len(self.leftovers()), 1)
        self.send(text="/reset")
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(self.app.session.state, State.IDLE)

    def test_cancel_during_await_cleans_up(self):
        self.press("swap")
        self.send(photo_id="src")
        self.send(text="/cancel")
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(self.app.session.state, State.CANCELLED)

    def test_status_and_help_do_not_leak_paths_or_ids(self):
        self.send(text="/status")
        self.send(text="/help")
        self.send(text="/start")
        txt = self.tg.texts()
        self.assertNotIn(self._tmp.name, txt)
        self.assertNotIn(str(ADMIN), txt)
        self.assertIn("private", txt)

    def test_upload_failure_reported_and_cleaned(self):
        self.tg.fail_upload = True
        self.begin()
        self.assertTrue(wait_for(lambda: self.app.session.state is State.FAILED))
        self.assertIn("uploading", self.tg.texts())
        self.assertEqual(self.leftovers(), [])

    def test_not_ready_blocks_start(self):
        self.app.swapper = None
        self.app._online.clear()
        self.press("swap")
        self.assertIn("not ready", self.tg.texts())
        self.assertEqual(self.app.session.state, State.IDLE)


class NoFaceTests(Base):
    swapper = FakeSwapper(error=NoFaceError("target"))

    def test_no_face_message_and_cleanup(self):
        self.begin()
        self.assertTrue(wait_for(lambda: self.app.session.state is State.FAILED))
        self.assertIn("Unable to detect a suitable face in the target", self.tg.texts())
        self.assertEqual(self.tg.images, [])
        self.assertEqual(self.leftovers(), [])


class AmbiguousTests(Base):
    swapper = FakeSwapper(error=AmbiguousFaceError("source"))

    def test_ambiguous_message(self):
        self.begin()
        self.assertTrue(wait_for(lambda: self.app.session.state is State.FAILED))
        self.assertIn("several faces", self.tg.texts())


class InternalErrorTests(Base):
    swapper = FakeSwapper(error=RuntimeError("secret /internal/path detail"))

    def test_internal_details_not_exposed(self):
        self.begin()
        self.assertTrue(wait_for(lambda: self.app.session.state is State.FAILED))
        self.assertNotIn("/internal/path", self.tg.texts())
        self.assertIn("server logs", self.tg.texts())


class TimeoutTests(Base):
    timeout = 1
    swapper = FakeSwapper(delay=1.6)  # non-interruptible work past the 1 s deadline

    def test_timeout_discards_late_result(self):
        self.begin()
        self.assertTrue(wait_for(lambda: "timed out" in self.tg.texts(), 4))
        self.assertTrue(wait_for(lambda: self.leftovers() == []))
        import time
        time.sleep(1.2)  # let the "native" call return
        self.assertEqual(self.tg.images, [])


class CancelRunningTests(Base):
    def setUp(self):
        import threading
        self.gate = threading.Event()
        self.swapper = FakeSwapper(gate=self.gate)
        super().setUp()

    def test_cancel_running_job(self):
        self.begin()
        self.assertTrue(wait_for(lambda: self.app.session.state is State.PROCESSING))
        self.send(text="/cancel")
        self.assertIn("cannot always be interrupted", self.tg.texts())
        self.gate.set()
        import time
        time.sleep(0.3)
        self.assertEqual(self.tg.images, [])
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(self.app.session.state, State.CANCELLED)


class ShutdownTests(Base):
    def setUp(self):
        import threading
        self.gate = threading.Event()
        self.swapper = FakeSwapper(gate=self.gate)
        super().setUp()

    def test_graceful_shutdown_with_active_job(self):
        self.begin()
        self.assertTrue(wait_for(lambda: self.app.session.state is State.PROCESSING))
        self.app.jobs.shutdown(0.2)
        self.assertIn("shutting down", self.tg.texts())
        self.assertEqual(self.leftovers(), [])
        self.gate.set()


if __name__ == "__main__":
    unittest.main()
