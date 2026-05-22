"""
ML model loading for SafeVision.

Models are loaded lazily (on first access) so the FastAPI server can start
accepting health-check requests before the heavy ONNX/PyTorch files are
loaded into memory.
"""

import logging
import threading

import torch

from app.config import settings

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Device detection — shared by both models
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = "cuda:0" if _CUDA_AVAILABLE else "cpu"
_ARCFACE_CTX = 0 if _CUDA_AVAILABLE else -1  # insightface: 0=GPU, -1=CPU

if _CUDA_AVAILABLE:
    gpu_name = torch.cuda.get_device_name(0)
    logger.info("GPU detected: %s — models will run on CUDA", gpu_name)
else:
    logger.info("No GPU detected — models will run on CPU")

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

                logger.info("Loading YOLO model: %s on %s", settings.YOLO_MODEL_PATH, _DEVICE)
                _yolo_model = YOLO(settings.YOLO_MODEL_PATH)
                _yolo_model.to(_DEVICE)  # explicitly move to GPU or CPU
                _yolo_model.fuse()
                logger.info("YOLO model loaded on %s.", _DEVICE)
    return _yolo_model


def get_arcface():
    """Return the ArcFace model, loading it on first call (thread-safe)."""
    global _arcface_model
    if _arcface_model is None:
        with _lock:
            if _arcface_model is None:
                from insightface.model_zoo import get_model

                logger.info("Loading ArcFace model: %s on %s", settings.ARCFACE_MODEL_PATH, _DEVICE)
                _arcface_model = get_model(settings.ARCFACE_MODEL_PATH)
                _arcface_model.prepare(ctx_id=_ARCFACE_CTX)  # 0=GPU, -1=CPU
                logger.info("ArcFace model loaded on %s.", _DEVICE)
    return _arcface_model
