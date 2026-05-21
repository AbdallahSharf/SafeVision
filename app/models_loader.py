"""
ML model loading for SafeVision.

Models are loaded lazily (on first access) so the FastAPI server can start
accepting health-check requests before the heavy ONNX/PyTorch files are
loaded into memory.
"""

import logging
import threading

from app.config import settings

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Lazy-loading singletons
# ---------------------------------------------------------------------------
_yolo_model = None
_arcface_model = None
_lock = threading.Lock()


def get_yolo():
    """Return the YOLO model, loading it on first call (thread-safe)."""
    global _yolo_model
    if _yolo_model is None:
        with _lock:
            if _yolo_model is None:  # double-check after acquiring lock
                from ultralytics import YOLO

                logger.info("Loading YOLO model: %s", settings.YOLO_MODEL_PATH)
                _yolo_model = YOLO(settings.YOLO_MODEL_PATH)
                _yolo_model.fuse()
                logger.info("YOLO model loaded.")
    return _yolo_model


def get_arcface():
    """Return the ArcFace model, loading it on first call (thread-safe)."""
    global _arcface_model
    if _arcface_model is None:
        with _lock:
            if _arcface_model is None:
                from insightface.model_zoo import get_model

                logger.info("Loading ArcFace model: %s", settings.ARCFACE_MODEL_PATH)
                _arcface_model = get_model(settings.ARCFACE_MODEL_PATH)
                _arcface_model.prepare(ctx_id=-1)
                logger.info("ArcFace model loaded.")
    return _arcface_model
