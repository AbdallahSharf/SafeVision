"""
Centralized configuration for SafeVision.

All settings are loaded from environment variables (with sensible defaults).
On import, ``python-dotenv`` reads a local ``.env`` file so that the same
code works both locally and inside a Docker container.
"""

import os
import sys
import logging
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env BEFORE reading any os.environ values
load_dotenv()

# Suppress noisy FFmpeg warnings (like HEVC reference frame drops)
if "OPENCV_FFMPEG_LOGLEVEL" not in os.environ:
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "16"  # Show errors and fatals only

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("safevision.log"),
    ],
)
logger = logging.getLogger("safevision")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _require_env(name: str) -> str:
    """Return the value of an env-var or exit with a clear error."""
    value = os.environ.get(name)
    if not value:
        logger.critical("Required environment variable %s is not set. Exiting.", name)
        sys.exit(1)
    return value


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Typed, validated application settings — instantiated once at startup."""

    # ── Database ──────────────────────────────────────────────────────────
    MONGO_URI: str = os.environ.get(
        "MONGO_URI",
        "mongodb+srv://admin:pass@cluster.mongodb.net/test?retryWrites=true&w=majority",
    )

    # ── Security & Alerts ─────────────────────────────────────────────────
    API_SECRET_KEY: str = os.environ.get("API_SECRET_KEY", "")
    FIREBASE_CREDENTIALS_PATH: str = os.environ.get("FIREBASE_CREDENTIALS_PATH", "")
    FCM_TOPIC: str = os.environ.get("FCM_TOPIC", "safevision-alerts")

    # ── Server ────────────────────────────────────────────────────────────
    PORT: int           = int(os.environ.get("PORT", "8080"))
    DISPLAY_OUTPUT: bool = os.environ.get("DISPLAY_OUTPUT", "false").lower() == "true"
    RTSP_URL:  str = field(default_factory=lambda: _require_env("RTSP_URL"))

    # ── Model paths ───────────────────────────────────────────────────────
    YOLO_MODEL_PATH:    str = os.environ.get("YOLO_MODEL_PATH",    "models/best.pt")
    ARCFACE_MODEL_PATH: str = os.environ.get("ARCFACE_MODEL_PATH", "models/w600k_r50.onnx")

    # ── Detection thresholds ──────────────────────────────────────────────
    YOLO_CONF_THRESHOLD: float = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.4"))
    BOX_CONF_THRESHOLD:  float = float(os.environ.get("BOX_CONF_THRESHOLD",  "0.6"))
    RECOG_THRESHOLD:     float = float(os.environ.get("RECOG_THRESHOLD",     "0.6"))

    # ── Frame / processing ────────────────────────────────────────────────
    FRAME_WIDTH:   int   = int(os.environ.get("FRAME_WIDTH",   "800"))
    FRAME_HEIGHT:  int   = int(os.environ.get("FRAME_HEIGHT",  "600"))
    FACE_SIZE:     int   = int(os.environ.get("FACE_SIZE",     "112"))
    FACE_MARGIN:   int   = int(os.environ.get("FACE_MARGIN",   "20"))
    IMGSZ:         int   = int(os.environ.get("IMGSZ",         "640"))
    QUEUE_SIZE:    int   = int(os.environ.get("QUEUE_SIZE",     "5"))
    HISTORY_LEN:   int   = int(os.environ.get("HISTORY_LEN",   "5"))
    QUEUE_TIMEOUT: float = float(os.environ.get("QUEUE_TIMEOUT", "2.0"))

    # ── MongoDB vector search ─────────────────────────────────────────────
    DB_NUM_CANDIDATES: int = int(os.environ.get("DB_NUM_CANDIDATES", "100"))
    DB_TOP_K:          int = int(os.environ.get("DB_TOP_K",          "5"))

    # ── Streaming ─────────────────────────────────────────────────────────
    # JPEG quality for the MJPEG stream (65 = ~35% smaller than 80, indistinguishable for surveillance)
    STREAM_JPEG_QUALITY: int = int(os.environ.get("STREAM_JPEG_QUALITY", "65"))

    # ── Face quality gate ─────────────────────────────────────────────────
    # Laplacian variance below this value means the face crop is too blurry to recognise reliably.
    # Typical values: crisp face ≈ 200+, motion blur ≈ 20–60.
    BLUR_THRESHOLD: float = float(os.environ.get("BLUR_THRESHOLD", "80.0"))

    # ── Frame skipping ────────────────────────────────────────────────────
    # Run YOLO detection only once every N frames; reuse boxes for the rest.
    # Higher = faster, but bounding boxes lag slightly on fast movement.
    DETECT_EVERY_N: int = int(os.environ.get("DETECT_EVERY_N", "3"))

    # ── Low-light enhancement ─────────────────────────────────────────────
    LOW_LIGHT_ENABLE:         bool  = os.environ.get("LOW_LIGHT_ENABLE", "true").lower() == "true"
    LOW_LIGHT_AUTO_THRESHOLD: int   = int(os.environ.get("LOW_LIGHT_AUTO_THRESHOLD", "80"))
    CLAHE_CLIP_LIMIT:         float = float(os.environ.get("CLAHE_CLIP_LIMIT",  "3.0"))
    CLAHE_TILE_SIZE:          int   = int(os.environ.get("CLAHE_TILE_SIZE",     "8"))
    DENOISE_STRENGTH:         int   = int(os.environ.get("DENOISE_STRENGTH",    "7"))


# Singleton — import ``settings`` everywhere
settings = Settings()
