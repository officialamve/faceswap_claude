import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.image_utils import (ImageValidationError, Limits, check_decodable, decode_image,
                             encode_result, inspect_image)
from app.storage import Workspace
from tests.helpers import image_bytes

L = Limits(max_bytes=5_000_000, max_pixels=1_000_000, max_dimension=2000)


class ImageTests(unittest.TestCase):
    def test_valid_formats(self):
        for fmt in ("JPEG", "PNG", "WEBP"):
            w, h, _ = inspect_image(image_bytes(fmt), L)
            self.assertEqual((w, h), (256, 192))
            self.assertEqual(decode_image(image_bytes(fmt), L).shape, (192, 256, 3))

    def test_invalid_inputs_rejected(self):
        for data in (b"", b"not an image at all", b"\xff\xd8\xff garbage", b"GIF89a" + b"\0" * 100):
            with self.assertRaises(ImageValidationError):
                inspect_image(data, L)

    def test_truncated_file(self):
        data = image_bytes("PNG")[:200]
        with self.assertRaises(ImageValidationError):
            decode_image(data, L)

    def test_mime_or_name_irrelevant_signature_decides(self):
        pdf = b"%PDF-1.4" + b"\0" * 500
        with self.assertRaises(ImageValidationError):
            inspect_image(pdf, L)

    def test_size_limit(self):
        with self.assertRaises(ImageValidationError):
            inspect_image(image_bytes("PNG"), Limits(1000, 10**6, 2000))

    def test_pixel_and_dimension_limits(self):
        big = io.BytesIO()
        Image.new("RGB", (3000, 3000), "white").save(big, "PNG")  # tiny file, 9M pixels
        with self.assertRaises(ImageValidationError):
            inspect_image(big.getvalue(), L)
        with self.assertRaises(ImageValidationError):
            inspect_image(image_bytes("PNG", size=(1900, 100)), Limits(5_000_000, 10**6, 1000))

    def test_too_small(self):
        with self.assertRaises(ImageValidationError):
            inspect_image(image_bytes("PNG", size=(32, 32)), L)

    def test_decompression_bomb_guard(self):
        big = io.BytesIO()
        Image.new("L", (6000, 6000)).save(big, "PNG")
        with self.assertRaises(ImageValidationError):
            decode_image(big.getvalue(), Limits(5_000_000, 1_000_000, 10_000))

    def test_exif_orientation_applied(self):
        arr = decode_image(image_bytes("JPEG", size=(256, 192), exif_orientation=6), L)
        self.assertEqual(arr.shape[:2], (256, 192))  # rotated: height<->width swapped

    def test_alpha_flattened_and_channel_order_bgr(self):
        buf = io.BytesIO()
        Image.new("RGBA", (128, 128), (255, 0, 0, 255)).save(buf, "PNG")
        arr = decode_image(buf.getvalue(), L)
        self.assertEqual(tuple(arr[0, 0]), (0, 0, 255))  # red in BGR

    def test_encode_result_photo_vs_document(self):
        small = encode_result(np.zeros((300, 400, 3), np.uint8))
        self.assertTrue(small.as_photo)
        check_decodable(small)
        large = encode_result(np.zeros((3000, 3000, 3), np.uint8))
        self.assertFalse(large.as_photo)  # >2560px => document to avoid Telegram downscaling
        extreme = encode_result(np.zeros((100, 2400, 3), np.uint8))
        self.assertFalse(extreme.as_photo)  # aspect ratio > 20

    def test_workspace_safety_and_cleanup(self):
        with tempfile.TemporaryDirectory() as t:
            ws = Workspace(Path(t) / "base")
            d = ws.create()
            ws.write_atomic(d, "source.bin", b"abc")
            self.assertEqual((d / "source.bin").read_bytes(), b"abc")
            with self.assertRaises(ValueError):
                ws.write_atomic(d, "../../evil", b"x")
            with self.assertRaises(ValueError):
                ws.remove(Path(t))
            stale = ws.base / "sess-old"
            stale.mkdir()
            self.assertEqual(ws.cleanup_stale(0), 1)   # stale removed, active kept
            self.assertTrue(d.exists())
            ws.remove(d)
            self.assertFalse(d.exists())


if __name__ == "__main__":
    unittest.main()
