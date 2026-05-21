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

from app.config import settings

# ---------------------------------------------------------------------------
# CLAHE instance (reused across calls)
# ---------------------------------------------------------------------------
clahe = cv2.createCLAHE(
    clipLimit=settings.CLAHE_CLIP_LIMIT,
    tileGridSize=(settings.CLAHE_TILE_SIZE, settings.CLAHE_TILE_SIZE),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_dark(frame: np.ndarray) -> bool:
    """Return ``True`` if the frame's mean brightness is below the threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < settings.LOW_LIGHT_AUTO_THRESHOLD


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
    l_ch = clahe.apply(l_ch)
    frame = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # Step 3 — gamma correction to lift shadows (gamma=0.6 → brighter midtones)
    inv_gamma = 1.0 / 0.6
    lut = np.array(
        [min(255, int((i / 255.0) ** inv_gamma * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    frame = cv2.LUT(frame, lut)

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
    l_ch = clahe.apply(l_ch)
    face = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    return face
