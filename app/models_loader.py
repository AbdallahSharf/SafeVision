"""
ML model loading for SafeVision.

Models are loaded lazily (on first access) so the FastAPI server can start
accepting health-check requests before the heavy ONNX/PyTorch files are
loaded into memory.

ArcFace is loaded directly via onnxruntime — no insightface dependency needed.
The w600k_r50.onnx model expects:
  - Input : float32 tensor (N, 3, 112, 112), normalised as (x - 127.5) / 128.0
  - Output: float32 tensor (N, 512) — face embedding
"""

import logging
import threading

import numpy as np
import onnxruntime as ort
import torch

from app.config import settings

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Device detection — shared by both models
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE = "cuda:0" if _CUDA_AVAILABLE else "cpu"

if _CUDA_AVAILABLE:
    gpu_name = torch.cuda.get_device_name(0)
    logger.info("GPU detected: %s — models will run on TensorRT/CUDA", gpu_name)
    _ORT_PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    logger.info("No GPU detected — models will run on CPU")
    _ORT_PROVIDERS = ["CPUExecutionProvider"]

# ---------------------------------------------------------------------------
# Minimal ArcFace wrapper (replaces insightface — same interface)
# ---------------------------------------------------------------------------
class ArcFaceONNX:
    """
    Runs the ArcFace ResNet-50 ONNX model directly via onnxruntime.
    Drop-in replacement for insightface's ArcFaceONNX that avoids the
    Cython compilation requirement.
    """

    def __init__(self, model_path: str):
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 2
        sess_opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=_ORT_PROVIDERS,
        )
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info(
            "ArcFace ONNX session created — provider: %s",
            self._session.get_providers()[0],
        )

    def get_feat(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        Compute a 512-d face embedding.

        Parameters
        ----------
        face_rgb : np.ndarray
            H×W×3 uint8 RGB image, already resized to (112, 112).

        Returns
        -------
        np.ndarray  shape (1, 512)
        """
        # (H, W, C) → (1, C, H, W)  +  normalise to [-1, 1]
        x = face_rgb.astype(np.float32)
        x = (x - 127.5) / 128.0
        x = x.transpose(2, 0, 1)[np.newaxis, ...]          # (1,3,112,112)
        return self._session.run([self._output_name], {self._input_name: x})[0]


# ---------------------------------------------------------------------------
# Lazy-loading singletons
# ---------------------------------------------------------------------------
_yolo_model    = None
_arcface_model = None
_lock          = threading.Lock()


def get_yolo():
    """Return the YOLO model, loading it on first call (thread-safe)."""
    global _yolo_model
    if _yolo_model is None:
        with _lock:
            if _yolo_model is None:
                from ultralytics import YOLO
                import os

                engine_path = settings.YOLO_MODEL_PATH.replace('.pt', '.engine')
                if os.path.exists(engine_path):
                    logger.info("Loading YOLO TensorRT engine: %s", engine_path)
                    _yolo_model = YOLO(engine_path, task='detect')
                else:
                    logger.info("Loading YOLO PyTorch model: %s on %s", settings.YOLO_MODEL_PATH, _DEVICE)
                    _yolo_model = YOLO(settings.YOLO_MODEL_PATH)
                    _yolo_model.to(_DEVICE)
                    _yolo_model.fuse()
                    
                logger.info("YOLO model loaded on %s.", _DEVICE)
    return _yolo_model


def get_arcface() -> ArcFaceONNX:
    """Return the ArcFace model, loading it on first call (thread-safe)."""
    global _arcface_model
    if _arcface_model is None:
        with _lock:
            if _arcface_model is None:
                logger.info(
                    "Loading ArcFace model: %s on %s",
                    settings.ARCFACE_MODEL_PATH,
                    _ORT_PROVIDERS[0],
                )
                _arcface_model = ArcFaceONNX(settings.ARCFACE_MODEL_PATH)
                logger.info("ArcFace model loaded.")
    return _arcface_model
