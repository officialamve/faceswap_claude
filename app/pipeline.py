"""Job body: read inputs -> decode -> swap -> encode -> verify. Runs in a worker thread."""
from __future__ import annotations

import time

from .face_swap import FaceSwapper
from .image_utils import EncodedResult, Limits, check_decodable, decode_image, encode_result
from .jobs import Job


def run_swap_job(job: Job, swapper: FaceSwapper, limits: Limits) -> EncodedResult:
    t0 = time.monotonic()
    src_bytes = (job.workdir / "source.bin").read_bytes()
    tgt_bytes = (job.workdir / "target.bin").read_bytes()
    job.checkpoint("read")
    source = decode_image(src_bytes, limits)
    target = decode_image(tgt_bytes, limits)
    del src_bytes, tgt_bytes
    job.timings["decode"] = round(time.monotonic() - t0, 3)
    out = swapper.swap(source, target, job.checkpoint)
    job.timings.update(out.timings)
    if out.image.shape != target.shape:
        raise RuntimeError("engine changed image dimensions")
    t1 = time.monotonic()
    encoded = encode_result(out.image)
    check_decodable(encoded)
    job.timings["encode"] = round(time.monotonic() - t1, 3)
    job.checkpoint("encoded")
    return encoded
