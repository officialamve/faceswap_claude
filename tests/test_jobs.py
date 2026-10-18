import threading
import time
import unittest
from pathlib import Path

from app.jobs import (DeadlineExceeded, DuplicateJobError, Job, JobManager, JobStatus,
                      ManagerClosedError, QueueFullError)
from tests.helpers import wait_for


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, job, status, payload):
        self.events.append((job.id, status, payload))

    def statuses(self):
        return [s for _, s, _ in self.events]


def job(i, chat=1):
    return Job(id=str(i), chat_id=chat, workdir=Path("/tmp"))


class JobTests(unittest.TestCase):
    def test_success(self):
        rec = Recorder()
        jm = JobManager(lambda j: "result", rec, 5)
        jm.submit(job(1))
        self.assertTrue(wait_for(lambda: rec.events))
        self.assertEqual(rec.events[0][1:], (JobStatus.SUCCEEDED, "result"))
        jm.shutdown(1)

    def test_failure_reported(self):
        rec = Recorder()

        def boom(j):
            raise ValueError("x")
        jm = JobManager(boom, rec, 5)
        jm.submit(job(1))
        self.assertTrue(wait_for(lambda: rec.events))
        self.assertEqual(rec.statuses(), [JobStatus.FAILED])
        jm.shutdown(1)

    def test_duplicate_and_queue_full(self):
        gate, rec = threading.Event(), Recorder()
        jm = JobManager(lambda j: gate.wait(5), rec, 5)
        jm.submit(job(1, chat=1))
        with self.assertRaises(DuplicateJobError):
            jm.submit(job(1, chat=2))      # same id
        with self.assertRaises(DuplicateJobError):
            jm.submit(job(2, chat=1))      # same chat already active
        wait_for(lambda: jm.snapshot()["busy_workers"] == 1)
        jm.submit(job(3, chat=3))          # fills the queue (max_queued=1)
        with self.assertRaises(QueueFullError):
            jm.submit(job(4, chat=4))
        gate.set()
        jm.shutdown(1)

    def test_deadline_via_checkpoint(self):
        rec = Recorder()

        def cooperative(j):
            while True:
                time.sleep(0.02)
                j.checkpoint("loop")
        jm = JobManager(cooperative, rec, 0.2)
        jm.submit(job(1))
        self.assertTrue(wait_for(lambda: rec.events))
        self.assertEqual(rec.statuses(), [JobStatus.TIMED_OUT])
        jm.shutdown(1)

    def test_deadline_with_uninterruptible_work_discards_late_result(self):
        rec = Recorder()
        jm = JobManager(lambda j: (time.sleep(0.6), "late")[1], rec, 0.2)
        jm.submit(job(1))
        self.assertTrue(wait_for(lambda: rec.events, 2))
        self.assertEqual(rec.statuses(), [JobStatus.TIMED_OUT])
        time.sleep(0.8)  # let the native call return
        self.assertEqual(rec.statuses(), [JobStatus.TIMED_OUT])  # no late SUCCEEDED
        jm.shutdown(1)

    def test_cancel_queued(self):
        gate, rec = threading.Event(), Recorder()
        jm = JobManager(lambda j: gate.wait(5), rec, 5)
        jm.submit(job(1, chat=1))
        wait_for(lambda: jm.snapshot()["busy_workers"] == 1)
        jm.submit(job(2, chat=2))
        self.assertEqual(jm.cancel("2"), "queued")
        gate.set()
        wait_for(lambda: len(rec.events) >= 2)
        self.assertEqual(sorted(s.value for s in rec.statuses()), ["cancelled", "succeeded"])
        jm.shutdown(1)

    def test_cancel_running_discards_result(self):
        gate, rec = threading.Event(), Recorder()
        jm = JobManager(lambda j: (gate.wait(5), "x")[1], rec, 5)
        jm.submit(job(1))
        wait_for(lambda: jm.snapshot()["busy_workers"] == 1)
        self.assertEqual(jm.cancel("1"), "running")
        gate.set()
        time.sleep(0.2)
        self.assertEqual(rec.statuses(), [JobStatus.CANCELLED])
        self.assertEqual(jm.cancel("1"), "none")
        jm.shutdown(1)

    def test_shutdown_cancels_active_job(self):
        gate, rec = threading.Event(), Recorder()
        jm = JobManager(lambda j: gate.wait(5), rec, 5)
        jm.submit(job(1))
        wait_for(lambda: jm.snapshot()["busy_workers"] == 1)
        jm.shutdown(0.2)
        self.assertEqual(rec.events[0][1:], (JobStatus.CANCELLED, "shutdown"))
        with self.assertRaises(ManagerClosedError):
            jm.submit(job(2))
        gate.set()

    def test_checkpoint_raises_after_deadline(self):
        j = job(1)
        j.deadline = time.monotonic() - 1
        with self.assertRaises(DeadlineExceeded):
            j.checkpoint("x")


if __name__ == "__main__":
    unittest.main()
