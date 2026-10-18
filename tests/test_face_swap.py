"""Geometry/decoding/selection tests (synthetic data) + model-file handling.
The ONNX sessions here are MOCKED; a real swap is exercised only by the integration test."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.config import ModelSpec
from app.face_swap import (ARCFACE_TEMPLATE, AmbiguousFaceError, Face, ModelInitError, NoFaceError,
                           OnnxFaceSwapper, ScrfdDetector, SwapModel, align_face, nms, paste_back,
                           select_face, umeyama)
from app.models import ModelIntegrityError, ModelMissingError, ensure_model


def face(x1, y1, x2, y2, score=0.9):
    return Face(np.array([x1, y1, x2, y2], np.float32), np.zeros((5, 2), np.float32), score)


class GeometryTests(unittest.TestCase):
    def test_umeyama_recovers_similarity(self):
        theta, s, t = 0.3, 1.7, np.array([12.0, -5.0])
        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        src = ARCFACE_TEMPLATE.astype(np.float64)
        dst = (s * src @ r.T) + t
        m = umeyama(src, dst)
        np.testing.assert_allclose(src @ m[:, :2].T + m[:, 2], dst, atol=1e-3)

    def test_align_face_shape(self):
        img = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
        crop, m = align_face(img, ARCFACE_TEMPLATE * 2 + 100, 128)
        self.assertEqual((crop.shape, m.shape), ((128, 128, 3), (2, 3)))

    def test_nms(self):
        d = np.array([[0, 0, 10, 10, .9], [1, 1, 10, 10, .8], [50, 50, 60, 60, .7]], np.float32)
        self.assertEqual(sorted(nms(d, 0.4)), [0, 2])

    def test_paste_back_changes_only_face_region_and_keeps_dimensions(self):
        target = np.full((300, 300, 3), 100, np.uint8)
        kps = ARCFACE_TEMPLATE * 1.0 + np.array([100, 100], np.float32)  # face near (100..180)
        _, m = align_face(target, kps, 128)
        fake = np.full((128, 128, 3), 220, np.uint8)
        out = paste_back(target, fake, m)
        self.assertEqual(out.shape, target.shape)
        self.assertTrue((out[0:20, 0:20] == 100).all())          # background untouched
        self.assertGreater(int(out[150, 140, 0]), 180)           # face centre replaced
        self.assertFalse(np.shares_memory(out, target))


class SelectionTests(unittest.TestCase):
    def test_no_face(self):
        with self.assertRaises(NoFaceError):
            select_face([], "source")
        with self.assertRaises(NoFaceError):
            select_face([face(0, 0, 10, 10)], "target")  # below minimum size

    def test_single_and_dominant(self):
        big, small = face(0, 0, 200, 200), face(300, 0, 350, 50)
        self.assertIs(select_face([small, big], "t"), big)

    def test_ambiguous(self):
        with self.assertRaises(AmbiguousFaceError):
            select_face([face(0, 0, 200, 200), face(300, 0, 480, 180)], "target")

    def test_deterministic_tiebreak(self):
        a, b = face(0, 0, 100, 100), face(500, 0, 560, 60)
        self.assertIs(select_face([b, a], "t"), a)


class FakeOut:
    def __init__(self, name):
        self.name = name


class FakeDetSession:
    """MOCKED SCRFD session emitting exactly one confident anchor at stride 8."""

    def __init__(self, n_out=9):
        self.n_out = n_out

    def get_outputs(self):
        return [FakeOut(f"o{i}") for i in range(self.n_out)]

    def get_inputs(self):
        return [FakeOut("input.1")]

    def run(self, names, feed):
        counts = [12800, 3200, 800]
        scores = [np.zeros((c, 1), np.float32) for c in counts]
        boxes = [np.zeros((c, 4), np.float32) for c in counts]
        kps = [np.zeros((c, 10), np.float32) for c in counts]
        k = (10 * 80 + 20) * 2          # grid (x=20, y=10), anchor 0 at stride 8 => centre (160, 80)
        scores[0][k] = 0.95
        boxes[0][k] = [4, 4, 4, 4]      # +-32px
        return scores + boxes + kps


class DetectorTests(unittest.TestCase):
    def test_decode_one_face(self):
        det = ScrfdDetector(FakeDetSession())
        faces = det.detect(np.zeros((640, 640, 3), np.uint8))
        self.assertEqual(len(faces), 1)
        np.testing.assert_allclose(faces[0].bbox, [128, 48, 192, 112], atol=1e-3)

    def test_scale_back_for_non_square_image(self):
        faces = ScrfdDetector(FakeDetSession()).detect(np.zeros((320, 1280, 3), np.uint8))
        # 1280x320 is resized by 0.5; boxes are mapped back by 2x
        np.testing.assert_allclose(faces[0].bbox, [256, 96, 384, 224], atol=1e-3)

    def test_incompatible_detector_rejected(self):
        with self.assertRaises(ModelInitError):
            ScrfdDetector(FakeDetSession(n_out=6))

    def test_swap_model_rejects_wrong_architecture(self):
        class S:
            def get_inputs(self):
                return [type("I", (), {"name": "a", "shape": [1, 3, 64, 64]})()]
        with self.assertRaises(ModelInitError):
            SwapModel(S(), np.zeros((512, 512)))


class EngineOrchestrationTests(unittest.TestCase):
    """MOCKED components: proves stage order, checkpoints and error propagation only."""

    def test_stages_and_checkpoints(self):
        class Det:
            def detect(self, img):
                return [face(100, 100, 300, 300)]

        class Emb:
            def embed(self, img, f):
                return np.ones(512, np.float32) / np.sqrt(512)

        class Swp:
            def latent(self, e):
                return e

            def run(self, crop, lat):
                return np.full((128, 128, 3), 250, np.uint8)
        f = Face(np.array([100, 100, 300, 300], np.float32), ARCFACE_TEMPLATE * 1.5 + 100, 0.9)
        Det.detect = lambda self, img: [f]
        eng = OnnxFaceSwapper(Det(), Emb(), Swp())
        stages = []
        out = eng.swap(np.zeros((400, 400, 3), np.uint8), np.zeros((400, 400, 3), np.uint8), stages.append)
        self.assertEqual(stages[0], "start")
        self.assertIn("blend", stages)
        self.assertEqual(out.image.shape, (400, 400, 3))
        self.assertGreater(int(out.image.max()), 0)

    def test_no_face_error_propagates(self):
        class Det:
            def detect(self, img):
                return []
        eng = OnnxFaceSwapper(Det(), None, None)
        with self.assertRaises(NoFaceError):
            eng.swap(np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200, 3), np.uint8), lambda s: None)


class ModelFileTests(unittest.TestCase):
    def test_missing_mismatch_ok(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.onnx"
            spec = ModelSpec("swapper", p, hashlib.sha256(b"abc").hexdigest())
            with self.assertRaises(ModelMissingError):
                ensure_model(spec)
            p.write_bytes(b"tampered")
            with self.assertRaises(ModelIntegrityError):
                ensure_model(spec)
            p.write_bytes(b"abc")
            self.assertEqual(ensure_model(spec), p)


@unittest.skipUnless(os.environ.get("FACESWAP_INTEGRATION") == "1",
                     "set FACESWAP_INTEGRATION=1 plus model env vars, SOURCE_IMAGE and TARGET_IMAGE")
class RealSwapIntegration(unittest.TestCase):
    def test_real_swap(self):
        from app.config import load_settings
        from app.face_swap import load_swapper
        from app.image_utils import Limits, decode_image, encode_result, check_decodable
        s = load_settings(os.environ)
        eng = load_swapper(s)
        lim = Limits(s.max_input_bytes, s.max_image_pixels, s.max_image_dimension)
        src = decode_image(Path(os.environ["SOURCE_IMAGE"]).read_bytes(), lim)
        tgt = decode_image(Path(os.environ["TARGET_IMAGE"]).read_bytes(), lim)
        out = eng.swap(src, tgt, lambda s_: None)
        enc = encode_result(out.image)
        check_decodable(enc)
        self.assertEqual(out.image.shape, tgt.shape)
        diff = np.abs(out.image.astype(int) - tgt.astype(int)).mean()
        self.assertGreater(diff, 0.05)  # tolerance-based, not byte equality


if __name__ == "__main__":
    unittest.main()
