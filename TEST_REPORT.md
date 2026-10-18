# Test report (build sandbox, 2026-10-18)

Environment: Python 3.12.3, numpy 2.4.4, Pillow 12.1.1, OpenCV 4.13.0, onnxruntime 1.24.4, requests 2.33.1.
The sandbox had **no network access**: pytest, ruff, aiohttp, onnx and model weights could not be installed or downloaded.

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -t .` | **81 run, 80 passed, 1 skipped, 0 failed** (re-run after the last code change) |
| `python -m compileall app scripts tests` | passed |
| Crude AST check for unused imports | none found (ruff itself was **not** run) |
| Real entrypoint smoke test (fake credentials, no Telegram access) | `/healthz` 200; `/readyz` 503 `initialization_failed`; webhook without secret 403; SIGTERM shutdown clean; token not present in logs; missing config -> exit code 2 |
| Skipped test | `RealSwapIntegration` - needs real model weights + images (`FACESWAP_INTEGRATION=1`) |

Mocked vs real: Telegram and the ONNX sessions are **mocked** in every test (`tests/helpers.py`, `FakeDetSession`).
Those tests prove orchestration, geometry (similarity transform recovery, paste-back, NMS), state, authorization,
queue/timeout/cancel/shutdown, cleanup and HTTP behaviour. They do **not** prove that a real swap works.

## NOT verified
* Any real face swap (no weights, no network). Output quality, speed and RAM are unmeasured.
* That my SCRFD / ArcFace / inswapper pre- and post-processing matches the real ONNX models bit-for-bit
  (implemented from the public InsightFace reference behaviour; first real run may need tuning).
* `pip install -r requirements.txt` on a clean machine, `onnx` compatibility, `docker build`, GitHub Actions run, ruff, pytest.
* A live Render deployment, `setWebhook` against Telegram, and current Telegram file-size limits (code assumes
  20 MB download, 10 MB photo / 50 MB document upload from the Bot API docs as I know them; re-check).
