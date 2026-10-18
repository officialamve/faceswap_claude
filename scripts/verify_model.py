"""Verify model files (checksum + loadability) and optionally run one real swap.

  python scripts/verify_model.py
  python scripts/verify_model.py --source a.jpg --target b.jpg --out result.jpg
  python scripts/verify_model.py --hash models/inswapper_128.onnx      # print a SHA-256
"""
import argparse
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ConfigError, load_settings  # noqa: E402
from app.face_swap import load_swapper  # noqa: E402
from app.image_utils import Limits, check_decodable, decode_image, encode_result  # noqa: E402
from app.models import ModelError, sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", metavar="FILE")
    ap.add_argument("--source")
    ap.add_argument("--target")
    ap.add_argument("--out", default="result.jpg")
    a = ap.parse_args()
    if a.hash:
        print(sha256_file(Path(a.hash)))
        return 0
    try:
        s = load_settings(os.environ)
    except ConfigError as exc:
        print("configuration invalid:", exc)
        return 2
    t0 = time.monotonic()
    try:
        eng = load_swapper(s)
    except (ModelError, Exception) as exc:  # noqa: BLE001 - report any init failure clearly
        print("MODEL INIT FAILED:", type(exc).__name__, exc)
        return 1
    print(f"models verified and loaded in {time.monotonic() - t0:.1f}s")
    if a.source and a.target:
        lim = Limits(s.max_input_bytes, s.max_image_pixels, s.max_image_dimension)
        src = decode_image(Path(a.source).read_bytes(), lim)
        tgt = decode_image(Path(a.target).read_bytes(), lim)
        t1 = time.monotonic()
        out = eng.swap(src, tgt, lambda _: None)
        enc = encode_result(out.image)
        check_decodable(enc)
        Path(a.out).write_bytes(enc.data)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"swap OK {tgt.shape[1]}x{tgt.shape[0]} in {time.monotonic() - t1:.1f}s "
              f"stages={out.timings} peak_rss~{rss:.0f}MB cpus={os.cpu_count()} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
