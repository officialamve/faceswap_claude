import unittest
from pathlib import Path

from app.state import InvalidTransition, Session, State, UpdateDeduper


class StateTests(unittest.TestCase):
    def test_happy_path(self):
        s = Session()
        for st in (State.AWAIT_SOURCE, State.AWAIT_TARGET, State.QUEUED, State.PROCESSING, State.COMPLETED):
            s.transition(st)
        s.transition(State.AWAIT_SOURCE)
        self.assertEqual(s.state, State.AWAIT_SOURCE)

    def test_invalid_transitions(self):
        s = Session()
        for bad in (State.AWAIT_TARGET, State.QUEUED, State.PROCESSING, State.COMPLETED):
            with self.assertRaises(InvalidTransition):
                s.transition(bad)
        s.transition(State.AWAIT_SOURCE)
        with self.assertRaises(InvalidTransition):  # target cannot precede source
            s.transition(State.QUEUED)

    def test_only_if_job_guard(self):
        s = Session()
        s.transition(State.AWAIT_SOURCE, workdir=Path("/x"))
        s.transition(State.AWAIT_TARGET)
        s.transition(State.QUEUED, job_id="a")
        self.assertFalse(s.transition(State.PROCESSING, only_if_job="other"))
        self.assertEqual(s.state, State.QUEUED)
        self.assertTrue(s.transition(State.PROCESSING, only_if_job="a"))
        self.assertTrue(s.busy)

    def test_reset(self):
        s = Session()
        s.transition(State.AWAIT_SOURCE, workdir=Path("/x"))
        wd, job = s.reset()
        self.assertEqual((wd, job, s.state), (Path("/x"), None, State.IDLE))

    def test_deduper(self):
        d = UpdateDeduper(capacity=2)
        self.assertFalse(d.seen(1))
        self.assertTrue(d.seen(1))
        d.seen(2)
        d.seen(3)  # evicts 1
        self.assertFalse(d.seen(1))


if __name__ == "__main__":
    unittest.main()
