"""Image validation (signature + real decoding), orientation, and result encoding."""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

# Telegram sendPhoto: <=10 MB, width+height <=10000, ratio <=20. sendDocument: <=50 MB.
PHOTO_MAX_BYTES = 10_000_000
DOCUMENT_MAX_BYTES = 50_000_000


class ImageValidationError(Exception):
    """Message is safe to show to the administrator."""


@dataclass(frozen=True)
class Limits:
    max_bytes: int
    max_pixels: int
    max_dimension: int


@dataclass(frozen=True)
class EncodedResult:
    data: bytes
    width: int
    height: int
    as_photo: bool
    mime: str = "image/jpeg"
    filename: str = "face_swap_result.jpg"


def sniff_format(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def _open(data: bytes, limits: Limits) -> Image.Image:
    if not data:
        raise ImageValidationError("The file is empty.")
    if len(data) > limits.max_bytes:
        raise ImageValidationError(f"The file is too large (limit {limits.max_bytes // 1_000_000} MB).")
    fmt = sniff_format(data)
    if fmt is None:
        raise ImageValidationError("Unsupported file type. Please send a JPEG, PNG or WebP image.")
    old = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = limits.max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data), formats=[fmt])
            w, h = img.size
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError("The image has too many pixels.") from None
    except Exception:
        raise ImageValidationError("The file could not be read as an image.") from None
    finally:
        Image.MAX_IMAGE_PIXELS = old
    if w * h > limits.max_pixels:
        raise ImageValidationError(f"The image has too many pixels (limit {limits.max_pixels:,}).")
    if max(w, h) > limits.max_dimension:
        raise ImageValidationError(f"The image is too large (max side {limits.max_dimension}px).")
    if min(w, h) < 64:
        raise ImageValidationError("The image is too small.")
    return img


def inspect_image(data: bytes, limits: Limits) -> tuple[int, int, str]:
    """Upload-time validation: signature, header limits, and an integrity check."""
    img = _open(data, limits)
    fmt = img.format or "?"
    try:
        img.verify()
    except Exception:
        raise ImageValidationError("The image file appears to be corrupted.") from None
    return img.size[0], img.size[1], fmt


def decode_image(data: bytes, limits: Limits) -> np.ndarray:
    """Fully decode to an upright 8-bit BGR array (EXIF orientation applied)."""
    img = _open(data, limits)
    try:
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            img = bg
        else:
            img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    except Exception:
        raise ImageValidationError("The image file appears to be corrupted.") from None
    return np.ascontiguousarray(arr[:, :, ::-1])  # RGB -> BGR


def encode_result(bgr: np.ndarray) -> EncodedResult:
    """Encode as high-quality JPEG; choose photo vs document by Telegram's limits."""
    h, w = bgr.shape[:2]
    for quality in (95, 90, 85):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        data = buf.tobytes()
        if len(data) <= DOCUMENT_MAX_BYTES:
            break
    else:
        raise ImageValidationError("The result is too large to send via Telegram.")
    # Telegram downsizes photos beyond ~2560px, so larger results go out as documents (full quality).
    as_photo = (len(data) <= PHOTO_MAX_BYTES and max(w, h) <= 2560
                and (w + h) <= 10_000 and max(w, h) / max(1, min(w, h)) <= 20)
    return EncodedResult(data=data, width=w, height=h, as_photo=as_photo)


def check_decodable(result: EncodedResult) -> None:
    """Prove the output is a valid image with the expected dimensions."""
    arr = cv2.imdecode(np.frombuffer(result.data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None or arr.shape[1] != result.width or arr.shape[0] != result.height:
        raise RuntimeError("output validation failed")
