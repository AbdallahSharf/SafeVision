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

if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        "|recv_buffer_size;65536|stimeout;5000000|timeout;5000000"
    )

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
    MONGO_URI: str = field(default_factory=lambda: _require_env("MONGO_URI"))

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
    BOX_CONF_THRESHOLD:  float = float(os.environ.get("BOX_CONF_THRESHOLD",  "0.4"))
    RECOG_THRESHOLD:     float = float(os.environ.get("RECOG_THRESHOLD",     "0.45"))

    # ── Frame / processing ────────────────────────────────────────────────
    FRAME_WIDTH:   int   = int(os.environ.get("FRAME_WIDTH",   "640"))
    FRAME_HEIGHT:  int   = int(os.environ.get("FRAME_HEIGHT",  "480"))
    FACE_SIZE:     int   = int(os.environ.get("FACE_SIZE",     "112"))
    FACE_MARGIN:   int   = int(os.environ.get("FACE_MARGIN",   "20"))
    IMGSZ:         int   = int(os.environ.get("IMGSZ",         "640"))
    QUEUE_SIZE:    int   = int(os.environ.get("QUEUE_SIZE",     "2"))
    HISTORY_LEN:   int   = int(os.environ.get("HISTORY_LEN",   "5"))
    QUEUE_TIMEOUT: float = float(os.environ.get("QUEUE_TIMEOUT", "2.0"))

    # ── MongoDB vector search ─────────────────────────────────────────────
    DB_NUM_CANDIDATES: int = int(os.environ.get("DB_NUM_CANDIDATES", "100"))
    DB_TOP_K:          int = int(os.environ.get("DB_TOP_K",          "5"))

    # ── Streaming ─────────────────────────────────────────────────────────
    # JPEG quality for the MJPEG stream
    STREAM_JPEG_QUALITY: int = int(os.environ.get("STREAM_JPEG_QUALITY", "70"))
    # Target FPS for the MJPEG stream output (caps encoding rate to avoid wasted CPU)
    TARGET_STREAM_FPS: int = int(os.environ.get("TARGET_STREAM_FPS", "15"))

    # ── Face quality gate ─────────────────────────────────────────────────
    # Laplacian variance below this value means the face crop is too blurry to recognise reliably.
    # Typical values: crisp face ≈ 200+, motion blur ≈ 20–60.
    BLUR_THRESHOLD: float = float(os.environ.get("BLUR_THRESHOLD", "80.0"))

    # ── Frame skipping ────────────────────────────────────────────────────
    # Run YOLO every N-th frame; on skipped frames the tracker predicts positions
    DETECT_EVERY_N_FRAMES: int = int(os.environ.get("DETECT_EVERY_N_FRAMES", "2"))
    
    # ── Low-light enhancement ─────────────────────────────────────────────
    # Low-light enhancement — disabled by default (costs 5ms CPU per face in bright rooms)
    LOW_LIGHT_ENABLE:         bool  = os.environ.get("LOW_LIGHT_ENABLE", "false").lower() == "true"
    LOW_LIGHT_AUTO_THRESHOLD: int   = int(os.environ.get("LOW_LIGHT_AUTO_THRESHOLD", "80"))
    CLAHE_CLIP_LIMIT:         float = float(os.environ.get("CLAHE_CLIP_LIMIT",  "3.0"))
    CLAHE_TILE_SIZE:          int   = int(os.environ.get("CLAHE_TILE_SIZE",     "8"))
    DENOISE_STRENGTH:         int   = int(os.environ.get("DENOISE_STRENGTH",    "0"))

    # ── GPU stream decode ─────────────────────────────────────────────────
    USE_GPU_DECODE: bool = os.environ.get("USE_GPU_DECODE", "true").lower() == "true"

    # ── FAISS local index ─────────────────────────────────────────────────
    # Interval (seconds) between periodic syncs of the FAISS index with MongoDB
    FAISS_SYNC_INTERVAL: int = int(os.environ.get("FAISS_SYNC_INTERVAL", "60"))

    # ── Alert rate limiting ───────────────────────────────────────────────
    # Minimum seconds between consecutive FCM alerts
    ALERT_COOLDOWN_SECONDS: int = int(os.environ.get("ALERT_COOLDOWN_SECONDS", "60"))


# Singleton — import ``settings`` everywhere
settings = Settings()
