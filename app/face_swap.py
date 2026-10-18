"""Local ONNX face-swap engine: SCRFD detection -> ArcFace embedding -> inswapper-style swap.

The engine targets the InsightFace-format ONNX models (det_10g, w600k_r50, inswapper_128).
Those weights are NOT freely licensed - see README. Everything here is plain NumPy/OpenCV +
ONNX Runtime (CPU)."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import cv2
import numpy as np

from .config import Settings
from .models import ensure_model

log = logging.getLogger(__name__)
Checkpoint = Callable[[str], None]

ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
MIN_FACE_SIDE = 40          # px; smaller detections are ignored as unreliable
AMBIGUITY_RATIO = 0.5       # 2nd-largest face >50% of largest area => ambiguous


class FaceSwapError(Exception):
    """Base for user-reportable engine errors."""


class NoFaceError(FaceSwapError):
    def __init__(self, role: str):
        super().__init__(f"no suitable face in {role} image")
        self.role = role


class AmbiguousFaceError(FaceSwapError):
    def __init__(self, role: str):
        super().__init__(f"multiple comparable faces in {role} image")
        self.role = role


class ModelInitError(Exception):
    pass


@dataclass
class Face:
    bbox: np.ndarray   # x1,y1,x2,y2
    kps: np.ndarray    # (5,2)
    score: float

    @property
    def area(self) -> float:
        return float(max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1]))


@dataclass
class SwapOutput:
    image: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)


class FaceSwapper(Protocol):
    def swap(self, source: np.ndarray, target: np.ndarray, checkpoint: Checkpoint) -> SwapOutput: ...


# ----------------------------------------------------------------- geometry helpers
def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (2x3) mapping src -> dst."""
    src, dst = src.astype(np.float64), dst.astype(np.float64)
    n = src.shape[0]
    sm, dm = src.mean(0), dst.mean(0)
    sd, dd = src - sm, dst - dm
    cov = dd.T @ sd / n
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    u, s, vt = np.linalg.svd(cov)
    rot = u @ np.diag(d) @ vt
    var = sd.var(axis=0).sum()
    scale = (s * d).sum() / var
    t = dm - scale * rot @ sm
    return np.hstack([scale * rot, t[:, None]]).astype(np.float32)


def align_face(img: np.ndarray, kps: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    template = ARCFACE_TEMPLATE * (size / 112.0)
    m = umeyama(kps, template)
    return cv2.warpAffine(img, m, (size, size), borderValue=0.0), m


def nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ovr <= thresh]
    return keep


def select_face(faces: list[Face], role: str) -> Face:
    """Deterministic choice: the largest face; reject if another is of comparable size."""
    usable = [f for f in faces
              if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= MIN_FACE_SIDE]
    if not usable:
        raise NoFaceError(role)
    usable.sort(key=lambda f: (-f.area, float(f.bbox[0]), float(f.bbox[1])))
    if len(usable) > 1 and usable[1].area > AMBIGUITY_RATIO * usable[0].area:
        raise AmbiguousFaceError(role)
    return usable[0]


def paste_back(target: np.ndarray, fake128: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Blend the 128x128 swapped crop into `target` using a feathered mask (ROI only)."""
    h, w = target.shape[:2]
    im = cv2.invertAffineTransform(m)
    corners = np.array([[0, 0], [128, 0], [128, 128], [0, 128]], np.float32)
    pts = corners @ im[:, :2].T + im[:, 2]
    margin = 0.25 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    x0, y0 = int(max(0, np.floor(pts[:, 0].min() - margin))), int(max(0, np.floor(pts[:, 1].min() - margin)))
    x1, y1 = int(min(w, np.ceil(pts[:, 0].max() + margin))), int(min(h, np.ceil(pts[:, 1].max() + margin)))
    if x1 <= x0 or y1 <= y0:
        raise FaceSwapError("face region outside image")
    im_roi = im.copy()
    im_roi[:, 2] -= (x0, y0)
    rw, rh = x1 - x0, y1 - y0
    fake = cv2.warpAffine(fake128, im_roi, (rw, rh), borderValue=0.0).astype(np.float32)
    white = cv2.warpAffine(np.full((128, 128), 255, np.float32), im_roi, (rw, rh), borderValue=0.0)
    white[white > 20] = 255
    ys, xs = np.where(white == 255)
    if ys.size == 0:
        raise FaceSwapError("empty blend mask")
    mask_size = int(np.sqrt(max(1, (ys.max() - ys.min())) * max(1, (xs.max() - xs.min()))))
    k = max(mask_size // 10, 10)
    white = cv2.erode(white, np.ones((k, k), np.uint8), iterations=1)
    k = max(mask_size // 20, 5)
    white = cv2.GaussianBlur(white, (2 * k + 1, 2 * k + 1), 0) / 255.0
    white = white[:, :, None]
    out = target.copy()
    roi = target[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(white * fake + (1 - white) * roi, 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------- models
class ScrfdDetector:
    def __init__(self, session, det_size=(640, 640), det_thresh=0.5, nms_thresh=0.4):
        outs = session.get_outputs()
        if len(outs) != 9:
            raise ModelInitError("detector is not an SCRFD model with keypoints (expected 9 outputs)")
        self.session, self.det_size = session, det_size
        self.det_thresh, self.nms_thresh = det_thresh, nms_thresh
        self.input_name = session.get_inputs()[0].name
        self.output_names = [o.name for o in outs]
        self.fmc, self.strides, self.num_anchors = 3, (8, 16, 32), 2

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        dw, dh = self.det_size
        if h / w > dh / dw:
            nh, nw = dh, int(dh / (h / w))
        else:
            nw, nh = dw, int(dw * (h / w))
        scale = nh / h
        canvas = np.zeros((dh, dw, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (dw, dh), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        all_s, all_b, all_k = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            bbox = outs[i + self.fmc].reshape(-1, 4) * stride
            kps = outs[i + 2 * self.fmc].reshape(-1, 10) * stride
            gh, gw = dh // stride, dw // stride
            if scores.size != gh * gw * self.num_anchors:
                raise ModelInitError("unexpected detector output shape")
            ys, xs = np.mgrid[:gh, :gw]
            centers = np.repeat((np.stack([xs, ys], -1).astype(np.float32).reshape(-1, 2) * stride),
                                self.num_anchors, axis=0)
            pos = np.where(scores >= self.det_thresh)[0]
            if pos.size == 0:
                continue
            c, b, k = centers[pos], bbox[pos], kps[pos]
            boxes = np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], 1)
            pts = np.stack([c[:, j % 2] + k[:, j] for j in range(10)], 1).reshape(-1, 5, 2)
            all_s.append(scores[pos]); all_b.append(boxes / scale); all_k.append(pts / scale)
        if not all_s:
            return []
        s, b, k = np.concatenate(all_s), np.concatenate(all_b), np.concatenate(all_k)
        keep = nms(np.hstack([b, s[:, None]]).astype(np.float32), self.nms_thresh)
        return [Face(b[i], k[i], float(s[i])) for i in keep]


class ArcFaceEmbedder:
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        shape = session.get_inputs()[0].shape
        if list(shape[2:]) != [112, 112]:
            raise ModelInitError("embedder input is not 112x112")

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        aligned, _ = align_face(img, face.kps, 112)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.session.run(None, {self.input_name: blob})[0].reshape(-1)
        if emb.size != 512:
            raise ModelInitError("embedder output is not 512-d")
        return emb / np.linalg.norm(emb)


class SwapModel:
    def __init__(self, session, emap: np.ndarray):
        ins = session.get_inputs()
        if len(ins) != 2 or list(ins[0].shape[2:]) != [128, 128] or emap.shape != (512, 512):
            raise ModelInitError("swap model is not an inswapper_128-compatible network")
        self.session, self.emap = session, emap.astype(np.float32)
        self.target_name, self.source_name = ins[0].name, ins[1].name
        self.out_name = session.get_outputs()[0].name

    def latent(self, normed_embedding: np.ndarray) -> np.ndarray:
        lat = normed_embedding.reshape(1, -1).astype(np.float32) @ self.emap
        return (lat / np.linalg.norm(lat)).astype(np.float32)

    def run(self, aligned_bgr: np.ndarray, latent: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(aligned_bgr, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        pred = self.session.run([self.out_name], {self.target_name: blob, self.source_name: latent})[0]
        img = np.clip(255 * pred.transpose(0, 2, 3, 1)[0], 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR


class OnnxFaceSwapper:
    def __init__(self, detector: ScrfdDetector, embedder: ArcFaceEmbedder, swapper: SwapModel):
        self.detector, self.embedder, self.swapper = detector, embedder, swapper

    def swap(self, source: np.ndarray, target: np.ndarray, checkpoint: Checkpoint) -> SwapOutput:
        t: dict[str, float] = {}
        clock = time.monotonic()

        def mark(name: str) -> None:
            nonlocal clock
            now = time.monotonic()
            t[name] = round(now - clock, 3)
            clock = now
            checkpoint(name)

        checkpoint("start")
        src_face = select_face(self.detector.detect(source), "source")
        mark("detect_source")
        emb = self.embedder.embed(source, src_face)
        mark("embed_source")
        tgt_face = select_face(self.detector.detect(target), "target")
        mark("detect_target")
        crop, m = align_face(target, tgt_face.kps, 128)
        fake = self.swapper.run(crop, self.swapper.latent(emb))
        mark("inference")
        out = paste_back(target, fake, m)
        mark("blend")
        return SwapOutput(out, t)


def load_swapper(settings: Settings) -> OnnxFaceSwapper:
    """Verify model files, create CPU ONNX sessions once, and return a ready engine."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ModelInitError("onnxruntime is not installed") from exc
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise ModelInitError("CPUExecutionProvider unavailable")
    paths = {s.name: ensure_model(s) for s in settings.models}  # raises ModelError subclasses
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    opts.enable_cpu_mem_arena = False  # lower peak RSS at a small speed cost

    def session(p):
        try:
            return ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ModelInitError(f"cannot load {p.name}: {type(exc).__name__}") from exc

    det, rec, swp = (session(paths[n]) for n in ("detector", "embedder", "swapper"))
    try:
        import onnx
        from onnx import numpy_helper
        graph = onnx.load(str(paths["swapper"]), load_external_data=False).graph
        emap = numpy_helper.to_array(graph.initializer[-1])
    except ImportError as exc:
        raise ModelInitError("the 'onnx' package is required to read the swap model's emap") from exc
    except Exception as exc:
        raise ModelInitError(f"cannot read swap model emap: {type(exc).__name__}") from exc
    return OnnxFaceSwapper(ScrfdDetector(det), ArcFaceEmbedder(rec), SwapModel(swp, emap))
