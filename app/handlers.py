"""Telegram update handling, conversation flow, and job callbacks."""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid

from .authorization import extract_actor, is_authorized
from .config import Settings
from .face_swap import AmbiguousFaceError, FaceSwapError, FaceSwapper, NoFaceError
from .image_utils import EncodedResult, ImageValidationError, Limits, inspect_image
from .jobs import (DuplicateJobError, Job, JobManager, JobStatus, ManagerClosedError,
                   QueueFullError)
from .pipeline import run_swap_job
from .state import InvalidTransition, Session, State, UpdateDeduper
from .storage import Workspace
from .telegram_api import FileTooLarge, TelegramClient, TelegramError, keyboard_json

log = logging.getLogger(__name__)

MENU = keyboard_json([[("Start Face Swap", "swap"), ("Cancel Current Job", "cancel")],
                      [("Help", "help"), ("Bot Status", "status")]])
CONSENT = ("You are responsible for having the rights and the consent of every person whose "
           "likeness you use. Do not use this bot for deception, harassment or intimate imagery.")
WELCOME = ("Welcome. This is a private face-swap bot: only its administrator can use it.\n\n"
           + CONSENT + "\n\nUse the buttons below.")
HELP = ("How it works:\n1. Press Start Face Swap.\n2. Send the SOURCE photo (the face to use).\n"
        "3. Send the TARGET photo (the picture whose face is replaced).\n\n"
        "Tips: one clear, front-facing face per photo. Send images as File/Document to avoid "
        "Telegram's recompression. If several faces of similar size are present the bot refuses "
        "rather than guessing.\n\nCommands: /start /help /cancel /status /reset\n\n" + CONSENT +
        "\n\nNote: deleting local files does not delete copies already delivered to or held by Telegram.")


class BotApp:
    def __init__(self, settings: Settings, tg: TelegramClient, workspace: Workspace) -> None:
        self.settings, self.tg, self.ws = settings, tg, workspace
        self.limits = Limits(settings.max_input_bytes, settings.max_image_pixels,
                             settings.max_image_dimension)
        self.session, self.dedupe = Session(), UpdateDeduper()
        self.swapper: FaceSwapper | None = None
        self.init_error: str | None = None
        self._online = threading.Event()
        self._updates: queue.Queue[dict | None] = queue.Queue(maxsize=100)
        self._thread = threading.Thread(target=self._dispatch_loop, name="dispatcher", daemon=True)
        self.jobs = JobManager(self._run_job, self._on_finished, settings.processing_timeout,
                               on_started=self._on_started, workers=settings.max_concurrent_jobs)

    # ------------------------------------------------------------ lifecycle / health
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.jobs.shutdown(5.0)
        self._updates.put(None)
        self._thread.join(5.0)

    def mark_online(self) -> None:
        self._online.set()

    def readiness(self) -> tuple[bool, str]:
        if self.init_error:
            return False, "initialization_failed"
        if self.swapper is None:
            return False, "loading_models"
        if not self._online.is_set():
            return False, "registering_webhook"
        return True, "ready"

    def submit_update(self, update: dict) -> bool:
        try:
            self._updates.put_nowait(update)
            return True
        except queue.Full:
            return False

    def _dispatch_loop(self) -> None:
        while True:
            update = self._updates.get()
            if update is None:
                return
            try:
                self.handle_update(update)
            except Exception:
                log.exception("unhandled error while handling update")
                _, chat_id, _ = extract_actor(update)
                if chat_id == self.settings.admin_id:
                    self._say(chat_id, "Something went wrong handling that message. Check server logs.")

    # ------------------------------------------------------------ helpers
    def _say(self, chat_id: int, text: str, menu: bool = False) -> None:
        try:
            self.tg.send_message(chat_id, text, MENU if menu else None)
        except TelegramError as exc:
            log.warning("sendMessage failed: %s", exc)

    # ------------------------------------------------------------ update entry point
    def handle_update(self, update: dict) -> None:
        if not is_authorized(update, self.settings.admin_id):
            self._reject(update)
            return
        uid = update.get("update_id")
        if isinstance(uid, int) and self.dedupe.seen(uid):
            log.info("duplicate update ignored")
            return
        _, chat_id, _ = extract_actor(update)
        assert chat_id is not None
        if "callback_query" in update:
            cq = update["callback_query"]
            try:
                self.tg.answer_callback(str(cq.get("id", "")))
            except TelegramError:
                pass
            self._action(chat_id, str(cq.get("data", "")))
            return
        msg = update["message"]
        if msg.get("photo") or msg.get("document"):
            self._on_media(chat_id, msg)
            return
        text = str(msg.get("text", "")).strip()
        if text.startswith("/"):
            self._action(chat_id, text.split()[0].split("@")[0][1:].lower())
        else:
            self._say(chat_id, "Use the buttons below or /help.", menu=True)

    def _reject(self, update: dict) -> None:
        log.warning("unauthorized update rejected")  # never log identities or content
        _, chat_id, ctype = extract_actor(update)
        cq = update.get("callback_query")
        try:
            if isinstance(cq, dict) and cq.get("id"):
                self.tg.answer_callback(str(cq["id"]), "Not authorized.")
            elif chat_id is not None and ctype == "private":
                self.tg.send_message(chat_id, "This is a private bot.")
        except TelegramError:
            pass

    def _action(self, chat_id: int, name: str) -> None:
        if name == "start":
            self._say(chat_id, WELCOME, menu=True)
        elif name == "help":
            self._say(chat_id, HELP, menu=True)
        elif name == "status":
            self._status(chat_id)
        elif name in ("swap",):
            self._start_swap(chat_id)
        elif name == "cancel":
            self._cancel(chat_id)
        elif name == "reset":
            self._reset(chat_id)
        else:
            self._say(chat_id, "Unknown command. Use /help.", menu=True)

    # ------------------------------------------------------------ commands
    def _status(self, chat_id: int) -> None:
        ready, reason = self.readiness()
        snap = self.jobs.snapshot()
        state = self.session.snapshot()[0]
        self._say(chat_id, f"Status: {'ready' if ready else 'not ready (' + reason + ')'}\n"
                           f"Conversation: {state.value}\nQueued jobs: {snap['queued']}\n"
                           f"Busy workers: {snap['busy_workers']}\n"
                           f"Processing limit: {self.settings.processing_timeout}s per job", menu=True)

    def _start_swap(self, chat_id: int) -> None:
        ready, reason = self.readiness()
        if not ready:
            self._say(chat_id, f"The bot is not ready yet ({reason}). Try again shortly.", menu=True)
            return
        if self.session.busy:
            self._say(chat_id, "A job is already queued or running. Wait for it or press Cancel.", menu=True)
            return
        old, _ = self.session.reset()  # discards any half-finished conversation
        self.ws.remove(old)
        wd = self.ws.create()
        self.session.transition(State.AWAIT_SOURCE, workdir=wd)
        self._say(chat_id, "Send the SOURCE photograph (the face to use).")

    def _cancel(self, chat_id: int) -> None:
        state, wd, job_id = self.session.snapshot()
        if state in (State.QUEUED, State.PROCESSING):
            result = self.jobs.cancel(job_id)
            if result == "running":
                self._say(chat_id, "Cancellation requested. Native inference cannot always be "
                                   "interrupted instantly; the worker may stay busy briefly, but the "
                                   "result will be discarded.", menu=True)
            elif result == "delivering":
                self._say(chat_id, "The result is already being delivered; it can no longer be cancelled.")
        elif state in (State.AWAIT_SOURCE, State.AWAIT_TARGET):
            self.session.transition(State.CANCELLED)
            self.ws.remove(wd)
            self._say(chat_id, "Cancelled. Temporary files removed.", menu=True)
        else:
            self._say(chat_id, "Nothing to cancel.", menu=True)

    def _reset(self, chat_id: int) -> None:
        wd, job_id = self.session.reset()
        if job_id:
            self.jobs.cancel(job_id)
        self.ws.remove(wd)
        self._say(chat_id, "State cleared and temporary files removed.", menu=True)

    # ------------------------------------------------------------ media
    def _pick_file(self, msg: dict) -> tuple[str, int | None] | None:
        photos = msg.get("photo")
        if photos:
            ok = [p for p in photos if (p.get("file_size") or 0) <= self.limits.max_bytes]
            best = max(ok or photos[:1], key=lambda p: (p.get("width", 0) * p.get("height", 0)))
            return best["file_id"], best.get("file_size")
        doc = msg.get("document") or {}
        if str(doc.get("mime_type", "")).startswith("image/") and doc.get("file_id"):
            return doc["file_id"], doc.get("file_size")
        return None

    def _on_media(self, chat_id: int, msg: dict) -> None:
        state, wd, _ = self.session.snapshot()
        if state not in (State.AWAIT_SOURCE, State.AWAIT_TARGET) or wd is None:
            self._say(chat_id, "Press Start Face Swap first.", menu=True)
            return
        picked = self._pick_file(msg)
        if picked is None:
            self._say(chat_id, "Please send a JPEG, PNG or WebP image (as photo or file).")
            return
        file_id, size = picked
        role = "source" if state is State.AWAIT_SOURCE else "target"
        if size and size > self.limits.max_bytes:
            self._say(chat_id, f"That file is too large (limit {self.limits.max_bytes // 1_000_000} MB).")
            return
        self._say(chat_id, f"Receiving {role} photograph.")
        t0 = time.monotonic()
        try:
            data = self.tg.download_file(file_id, self.limits.max_bytes)
        except FileTooLarge:
            self._say(chat_id, "That file is too large to download.")
            return
        except TelegramError as exc:
            log.warning("download failed: %s", exc)
            self._say(chat_id, "Could not download that file from Telegram. Please send it again.")
            return
        try:
            inspect_image(data, self.limits)
        except ImageValidationError as exc:
            self._say(chat_id, f"The {role} photograph was rejected: {exc}")
            return
        download_s = time.monotonic() - t0
        self.ws.write_atomic(wd, f"{role}.bin", data)
        del data
        if role == "source":
            self.session.transition(State.AWAIT_TARGET)
            self._say(chat_id, "Source photograph received. Waiting for target photograph.")
        else:
            self._enqueue(chat_id, wd, download_s)

    def _enqueue(self, chat_id: int, wd, download_s: float) -> None:
        job = Job(id=uuid.uuid4().hex[:12], chat_id=chat_id, workdir=wd)
        job.timings["download"] = round(download_s, 3)
        try:
            self.session.transition(State.QUEUED, job_id=job.id)
        except InvalidTransition:
            return
        try:
            self.jobs.submit(job)
        except (DuplicateJobError, QueueFullError, ManagerClosedError) as exc:
            self.session.transition(State.FAILED)
            self.ws.remove(wd)
            self._say(chat_id, f"Could not queue the job ({type(exc).__name__}). Try again later.", menu=True)
            return
        self._say(chat_id, f"Target photograph received. Job {job.id} queued.")

    # ------------------------------------------------------------ job callbacks (worker threads)
    def _run_job(self, job: Job) -> EncodedResult:
        assert self.swapper is not None
        return run_swap_job(job, self.swapper, self.limits)

    def _on_started(self, job: Job) -> None:
        try:
            self.session.transition(State.PROCESSING, only_if_job=job.id)
        except InvalidTransition:
            pass
        self._say(job.chat_id, "Processing image.")

    def _on_finished(self, job: Job, status: JobStatus, payload) -> None:
        extra = {"job_id": job.id}
        final = State.FAILED
        try:
            if status is JobStatus.SUCCEEDED:
                final = State.COMPLETED if self._deliver(job, payload) else State.FAILED
            elif status is JobStatus.TIMED_OUT:
                log.error("job timed out", extra=extra)
                self._say(job.chat_id, f"Processing timed out after {self.settings.processing_timeout} s. "
                                       "Any late result is discarded. Try smaller images.", menu=True)
            elif status is JobStatus.CANCELLED:
                final = State.CANCELLED
                self._say(job.chat_id, "Service is shutting down; the job was cancelled."
                          if payload == "shutdown" else "Job cancelled. Temporary files removed.", menu=True)
            else:
                if isinstance(payload, (FaceSwapError, ImageValidationError)):
                    log.warning("job rejected input: %s", type(payload).__name__, extra=extra)
                else:
                    log.error("job failed: %s", type(payload).__name__, extra=extra, exc_info=payload)
                self._say(job.chat_id, self._error_text(job, payload), menu=True)
        finally:
            self.ws.remove(job.workdir)
            try:
                self.session.transition(final, only_if_job=job.id)
            except InvalidTransition:
                pass

    def _deliver(self, job: Job, result: EncodedResult) -> bool:
        t0 = time.monotonic()
        total = time.monotonic() - (job.started or t0)
        caption = f"Face swap completed in {total:.1f} s (job {job.id})."
        try:
            self.tg.send_image(job.chat_id, result.data, result.filename, result.mime, caption,
                               result.as_photo)
        except TelegramError as exc:
            log.error("upload failed: %s", exc, extra={"job_id": job.id})
            self._say(job.chat_id, "The result was created but uploading it to Telegram failed.", menu=True)
            return False
        job.timings["upload"] = round(time.monotonic() - t0, 3)
        log.info("job delivered timings=%s", job.timings, extra={"job_id": job.id})
        self._say(job.chat_id, "Done. Start another swap any time.", menu=True)
        return True

    @staticmethod
    def _error_text(job: Job, exc) -> str:
        if isinstance(exc, NoFaceError):
            return (f"Unable to detect a suitable face in the {exc.role} photograph. "
                    "Please send a sharper, well-lit, front-facing photo.")
        if isinstance(exc, AmbiguousFaceError):
            return (f"The {exc.role} photograph has several faces of similar size, so I won't guess. "
                    "Please crop to one face.")
        if isinstance(exc, ImageValidationError):
            return f"Processing failed: {exc}"
        if isinstance(exc, MemoryError):
            return "Processing failed: the server ran out of memory. Try smaller images."
        return f"Processing failed. Details are in the server logs (job {job.id})."
