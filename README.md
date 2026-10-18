# Private Telegram face-swap bot (local ONNX inference, Render)

A single-administrator Telegram bot. You send a **source** photo (the face to use) and a **target** photo; the bot
replaces the main face in the target using models running **inside your own process** (no paid inference API) and sends
the result back. Pure-Python stdlib HTTP server + `requests`, one worker thread, no database.

> ## READ FIRST - two blockers you must resolve
> 1. **Model licence.** The only widely used, good-quality local pipeline is InsightFace's `det_10g` + `w600k_r50` +
>    `inswapper_128`. InsightFace's repository states its pretrained models are for **non-commercial research only**, and
>    (as of its 2025-11-24 notice) tells users of the *inswapper* models to **contact InsightFace for licensing**.
>    I therefore **do not bundle, link or auto-download any weights**. You must obtain them lawfully yourself and host
>    them where only you can fetch them (or place them in `models/`). I know of no permissively licensed drop-in
>    alternative that I could verify; if you need commercial or clearly unrestricted use, get a licence or swap the engine.
> 2. **Render Free cannot run this.** Render Free is 512 MB RAM / 0.1 CPU; the three models are roughly 740 MB on disk
>    (my estimate from the public InsightFace releases - verify against your files) and `onnx` loads the 550 MB swap model
>    once to read one table at start-up. Use at least **Standard (2 GB, 1 CPU, ~$25/month per Render's published pricing)**.
>    `render.yaml` therefore says `plan: standard`. Free also sleeps after 15 min idle and has no persistent disk.
>    I added no keep-alive traffic to evade that.

## Status of verification
See `TEST_REPORT.md`: 80 mocked/unit tests pass; **no real swap was run** (no network/weights in my build environment).
Treat the first deployment as the real acceptance test. Nothing here promises speed or quality; the 600 s limit is a
*maximum*.

## How it works (high level)
1. SCRFD detects faces and 5 landmarks. If none is usable the job fails with a clear message; if a second face is
   >50% the area of the largest, the job is **rejected as ambiguous** (never guesses). Otherwise the largest face is used.
2. ArcFace turns the source face into a 512-d identity vector.
3. The target face is aligned to a 128x128 crop (similarity transform from landmarks); the swap network renders the new
   face; it is warped back and blended with an eroded, feathered mask. Background and image size are preserved.
4. The result is JPEG-encoded (q95), decoded again to prove it is valid, then sent as a photo (<=2560 px, <=10 MB) or as
   a document (larger, to avoid Telegram downscaling).

## Limits that remain even without a paid API
Local inference still costs CPU/RAM/time on your Render instance (and its monthly price); Telegram, Render, network
conditions and model licences impose their own limits. The app has **no daily quota**. Jobs run one at a time
(`MAX_CONCURRENT_JOBS=1`); the queue is in memory, so queued/running jobs **do not survive a restart** (the bot tells you
"if a job was running before a restart, it was lost").

**Deadline semantics.** Each job has a 600 s monotonic deadline checked between stages. Native ONNX calls cannot be
interrupted: on timeout or `/cancel` the job is declared final immediately, the late result is discarded and never sent,
but the worker thread stays busy until the native call returns.

## Requirements
Python 3.12; ~2 GB RAM minimum for the full model set; CPU-only (`onnxruntime`). No GPU needed. Cold start = process
start + (optional model download) + checksum + session creation; the port is bound immediately, `/readyz` turns 200 only
when models are loaded **and** the webhook is registered.

## Setup (Windows PowerShell)
```powershell
# 1. Bot token: in Telegram talk to @BotFather -> /newbot -> copy the token.
# 2. Your numeric Telegram ID: message @userinfobot (third-party) OR message your own bot, then:
#    (token ends up in shell history; clear it afterwards. Works only while NO webhook is set.)
Invoke-RestMethod "https://api.telegram.org/bot<TOKEN>/getUpdates" | ConvertTo-Json -Depth 6   # look for "from": {"id": ...}

# 3. Local checks
git clone <this repo> ; cd face-swap-telegram-bot
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q

# 4. Hash your model files (put them in .\models\, never commit them)
python scripts\verify_model.py --hash models\inswapper_128.onnx      # repeat for det_10g.onnx, w600k_r50.onnx

# 5. Create the GitHub repo (empty, private) and push
git init ; git add . ; git status          # confirm no .env / models are listed
git commit -m "Initial commit" ; git branch -M main
git remote add origin https://github.com/<you>/face-swap-telegram-bot.git
git push -u origin main
```
Linux/macOS equivalent: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt && pytest -q`.

## Deploy on Render
1. Render dashboard -> **New -> Blueprint** -> select the repo (reads `render.yaml`, Docker, `standard` plan).
2. Fill the prompted environment variables:

| Variable | Required | Notes |
|---|---|---|
| `BOT_TOKEN` | yes | from BotFather |
| `ADMIN_TELEGRAM_ID` | yes | numeric, positive |
| `WEBHOOK_SECRET` | yes | 16-256 chars `A-Za-z0-9_-` (generate yourself) |
| `WEBHOOK_URL` | no on Render | defaults to Render's `RENDER_EXTERNAL_URL`; must be https |
| `DET_/REC_/SWAP_MODEL_SHA256` | yes | 64-hex SHA-256 of each model file (integrity gate) |
| `DET_/REC_/SWAP_MODEL_URL` | yes on Render* | https URL **you** control; downloaded at start, verified, re-downloaded after every restart because the disk is ephemeral |
| `PROCESSING_TIMEOUT_SECONDS` | no | 1-600, default 600 |
| `DOWNLOAD_TIMEOUT_SECONDS`, `MAX_INPUT_BYTES` (<=20 MB), `MAX_IMAGE_PIXELS`, `MAX_IMAGE_DIMENSION`, `MAX_CONCURRENT_JOBS`, `TEMP_DIR`, `MODEL_DIR`, `LOG_LEVEL` | no | see `.env.example` |
| `PORT` | no | Render injects it; server binds `0.0.0.0:$PORT` |

\* The repo cannot contain the weights; alternatively bake them into your own private image.
Polling mode is **not** implemented, so `WEBHOOK_*` variables are always required.

3. The service registers the webhook itself (with the secret token) after models load. Check:
```powershell
Invoke-RestMethod https://<service>.onrender.com/healthz      # {"status":"ok"}  liveness
Invoke-RestMethod https://<service>.onrender.com/readyz       # 503 until models+webhook are ready, then {"ready":true}
```
4. **Logs:** Render dashboard -> service -> Logs (key=value lines with `job=<id>`; no image data, tokens or URLs).
5. **Update:** `git commit` + `git push`; `autoDeploy` is off, so press *Manual Deploy* (or set it to `true`).

### Troubleshooting
* `Configuration error: ...` in logs, exit code 2: fix the named variable.
* `readyz` stays 503 `initialization_failed`: read the log line - `model: ... not found`, `checksum mismatch`,
  `incompatible`, or `telegram: getMe` (bad token). The service intentionally does **not** retry or crash-loop; fix and redeploy.
* Out-of-memory / restarts during start-up: instance too small - upgrade the plan.
* "Processing timed out": use smaller images (CPU time grows with size), or a larger instance.
* `onnx` import problems: only needed to read the swap model's `emap`; pin another `onnx` version in `requirements.txt`.

## Using it / acceptance checklist
- [ ] `/readyz` is 200. [ ] `/start` from your account shows the 4 buttons; from any other account only "This is a private bot."
- [ ] Start Face Swap -> source -> target -> result arrives; caption shows duration.
- [ ] A photo without a face gives "Unable to detect a suitable face". [ ] A group photo gives the "several faces" refusal.
- [ ] `/cancel` and `/reset` clear state; Render logs show no tokens. [ ] Redeploy mid-idle: bot announces it is online.
Send photos **as files** for best quality. Record real timings from the logs (`timings=` line) before relying on them.

## Deployment checklist
Obtain licensed weights -> compute SHA-256s -> host them privately over https -> push repo -> create Blueprint (Standard) ->
set env vars -> wait for `/readyz` -> run the acceptance checklist.

## Model licensing and attribution
Code in this repo: MIT. Model weights: **not covered, not included.** InsightFace (https://github.com/deepinsight/insightface):
code MIT; pretrained models non-commercial research only; inswapper requires a licence from InsightFace. Credit InsightFace
(SCRFD, ArcFace/`w600k_r50`, inswapper). You alone are responsible for the rights and consent of anyone depicted.

## Layout
`app/` source (`config`, `authorization`, `state`, `storage`, `image_utils`, `models`, `face_swap`, `jobs`, `pipeline`,
`telegram_api`, `handlers`, `health`, `main`); `tests/`; `scripts/check_environment.py`, `scripts/verify_model.py`;
`Dockerfile`, `render.yaml`, `.github/workflows/tests.yml`. (No `bot.py`: its role is split between `telegram_api` and `handlers`.)
