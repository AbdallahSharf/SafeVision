"""
Low-light image enhancement for SafeVision.

Two-level pipeline:
* **Frame-level** (``enhance_frame``) — applied to the full frame *before*
  YOLO detection.  Only activates when the frame is detected as dark.
* **Face-level** (``enhance_face``) — applied to each cropped face *before*
  ArcFace embedding.  Always runs when low-light mode is enabled because
  individual crops can be locally dark even if the overall frame is bright.
"""

import cv2
import numpy as np
import threading

from app.config import settings

# ---------------------------------------------------------------------------
# CLAHE instance (reused across calls)
# ---------------------------------------------------------------------------
_clahe = cv2.createCLAHE(
    clipLimit=settings.CLAHE_CLIP_LIMIT,
    tileGridSize=(settings.CLAHE_TILE_SIZE, settings.CLAHE_TILE_SIZE),
)
_clahe_lock = threading.Lock()


def _apply_clahe(channel: np.ndarray) -> np.ndarray:
    """Thread-safe CLAHE apply."""
    with _clahe_lock:
        return _clahe.apply(channel)

# Precomputed Gamma Lookup Table (gamma=0.6) for fast shadow lifting
_GAMMA_LUT = np.array(
    [min(255, int((i / 255.0) ** (1.0 / 0.6) * 255)) for i in range(256)],
    dtype=np.uint8,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dark-detection cache — recomputed every N frames, sub-sampled for speed
# ---------------------------------------------------------------------------
_DARK_RECHECK_INTERVAL = 10   # recheck brightness every 10 frames (~333 ms at 30 fps)
_dark_cache: dict = {"result": False, "counter": 0}


def is_dark(frame: np.ndarray) -> bool:
    """
    Return ``True`` if the frame's mean brightness is below the threshold.

    Uses a sub-sampled green-channel average (every 8th pixel → 1/64 of
    total pixels) and caches the result for ``_DARK_RECHECK_INTERVAL`` frames.
    This reduces the cost from ~3 ms (full cvtColor) to ~15 µs per call.
    """
    _dark_cache["counter"] += 1
    if _dark_cache["counter"] % _DARK_RECHECK_INTERVAL != 0:
        return _dark_cache["result"]
    # Green channel (index 1) approximates perceptual luminance well
    sampled = frame[::8, ::8, 1]    # shape ≈ (75, 100) for an 800×600 frame
    _dark_cache["result"] = float(sampled.mean()) < settings.LOW_LIGHT_AUTO_THRESHOLD
    return _dark_cache["result"]


# ---------------------------------------------------------------------------
# Frame-level enhancement
# ---------------------------------------------------------------------------
def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full-frame low-light enhancement applied **before** YOLO detection.

    Pipeline: optional denoise → CLAHE on L channel (LAB) → gamma lift.
    Only activates when ``LOW_LIGHT_ENABLE=true`` **and** the frame is dark.
    """
    if not settings.LOW_LIGHT_ENABLE or not is_dark(frame):
        return frame

    # Step 1 — fast denoise (skip if DENOISE_STRENGTH=0 to preserve FPS)
    if settings.DENOISE_STRENGTH > 0:
        frame = cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            h=settings.DENOISE_STRENGTH,
            hColor=settings.DENOISE_STRENGTH,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # Step 2 — CLAHE on L channel of LAB (boosts contrast without blowing color)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = _apply_clahe(l_ch)
    frame = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # Step 3 — gamma correction to lift shadows (gamma=0.6 → brighter midtones)
    frame = cv2.LUT(frame, _GAMMA_LUT)

    return frame


# ---------------------------------------------------------------------------
# Face-crop enhancement
# ---------------------------------------------------------------------------
def enhance_face(face: np.ndarray) -> np.ndarray:
    """
    Face-crop enhancement applied **before** ArcFace embedding.

    CLAHE on L channel only — keeps colour channels intact for ArcFace.
    """
    if not settings.LOW_LIGHT_ENABLE:
        return face

    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = _apply_clahe(l_ch)
    face = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    return face
