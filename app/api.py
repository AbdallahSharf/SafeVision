"""
FastAPI application for SafeVision.

Phase 2 refactor: the single background processing loop is now split into
three concurrent threads connected by bounded queues:

  Thread 1 — _reader_loop()
      Reads raw frames from the RTSP VideoStream and puts them into
      _rtsp_queue.  Drops the oldest frame when the queue is full so the
      system always works with the freshest available image.

  Thread 2 — _detector_loop()
      Pops raw frames from _rtsp_queue, runs Stage 1 of FrameProcessor
      (YOLO with frame-skipping), and pushes (frame, boxes) pairs into
      _detect_queue.

  Thread 3 — _recognizer_loop()
      Pops (frame, boxes) from _detect_queue, runs Stage 2 of
      FrameProcessor (ArcFace + tracker + annotation), JPEG-encodes the
      result, and stores it in _latest_frame for the MJPEG endpoint.

This pipeline means all three stages run simultaneously on different CPU
cores, roughly doubling throughput compared to the serial approach.

Endpoints
---------
GET  /          → API information
GET  /health    → Health check (for monitoring / load balancers)
GET  /status    → System metrics (FPS, stream state, DB face count)
GET  /stream    → Live MJPEG video stream with face recognition overlays
GET  /faces     → Recently recognised faces (JSON)
"""

from app.config import settings
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.database import faces_collection
from app.stream import VideoStream
from app.processor import FrameProcessor
from app.recognition import reload_thresholds

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Shared state (populated on startup)
# ---------------------------------------------------------------------------
_video_stream: VideoStream | None = None
_processor: FrameProcessor | None = None
_start_time: float = 0.0

# Inter-thread queues
_rtsp_queue: queue.Queue = queue.Queue(maxsize=3)     # raw BGR frames
_detect_queue: queue.Queue = queue.Queue(maxsize=3)   # (preprocessed_frame, boxes)

# Latest JPEG bytes served to MJPEG clients
_latest_frame: bytes = b""
_frame_lock = threading.Lock()

# Thread handles (daemon=True so they die when the main process exits)
_reader_thread: threading.Thread | None = None
_detector_thread: threading.Thread | None = None
_recognizer_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Pipeline threads
# ---------------------------------------------------------------------------
def _reader_loop() -> None:
    """
    Thread 1 — RTSP reader.

    Continuously reads frames from VideoStream and pushes them into
    _rtsp_queue.  When the queue is full, the oldest frame is dropped so
    we always keep the most recent one.
    """
    logger.info("Reader thread started.")
    while _video_stream and _video_stream.is_alive():
        frame = _video_stream.read()
        if frame is None:
            continue
        # Drop oldest frame if consumer is too slow — keep stream fresh
        if _rtsp_queue.full():
            try:
                _rtsp_queue.get_nowait()
            except queue.Empty:
                pass
        _rtsp_queue.put(frame)
    logger.info("Reader thread exited.")


def _detector_loop() -> None:
    """
    Thread 2 — YOLO detector.

    Pops raw frames from _rtsp_queue, runs Stage 1 of FrameProcessor
    (preprocessing + YOLO with frame-skipping), and pushes results into
    _detect_queue.
    """
    logger.info("Detector thread started.")
    while True:
        try:
            raw_frame = _rtsp_queue.get(timeout=2.0)
        except queue.Empty:
            if not (_video_stream and _video_stream.is_alive()):
                break
            continue

        preprocessed, boxes = _processor.detect(raw_frame)

        if _detect_queue.full():
            try:
                _detect_queue.get_nowait()
            except queue.Empty:
                pass
        _detect_queue.put((preprocessed, boxes))

    logger.info("Detector thread exited.")


def _recognizer_loop() -> None:
    """
    Thread 3 — ArcFace recognizer + annotator.

    Pops (frame, boxes) from _detect_queue, runs Stage 2 of
    FrameProcessor (ArcFace + ByteTrack smoothing + drawing), encodes
    the result as JPEG, and stores it in _latest_frame.
    """
    global _latest_frame
    logger.info("Recognizer thread started.")
    while True:
        try:
            frame, boxes = _detect_queue.get(timeout=2.0)
        except queue.Empty:
            if not (_video_stream and _video_stream.is_alive()):
                break
            continue

        result = _processor.recognize_and_annotate(frame, boxes)

        _, jpeg = cv2.imencode(
            ".jpg", result.annotated,
            [cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_JPEG_QUALITY],
        )
        with _frame_lock:
            _latest_frame = jpeg.tobytes()

    logger.info("Recognizer thread exited.")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the 3-thread pipeline on startup; clean up on shutdown."""
    global _video_stream, _processor, _start_time
    global _reader_thread, _detector_thread, _recognizer_thread

    _start_time = time.time()

    logger.info("Starting VideoStream …")
    _video_stream = VideoStream()
    _processor = FrameProcessor()

    # Start all three pipeline threads
    _reader_thread = threading.Thread(target=_reader_loop, daemon=True, name="sv-reader")
    _detector_thread = threading.Thread(target=_detector_loop, daemon=True, name="sv-detector")
    _recognizer_thread = threading.Thread(target=_recognizer_loop, daemon=True, name="sv-recognizer")

    _reader_thread.start()
    _detector_thread.start()
    _recognizer_thread.start()

    logger.info("SafeVision API is ready — 3-thread pipeline active.")
    yield  # ── app is running ──

    # Shutdown — VideoStream.stop() signals the reader loop to exit
    logger.info("Shutting down …")
    if _video_stream:
        _video_stream.stop()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SafeVision API",
    description="Real-time face recognition security system",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow mobile apps to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------
def _placeholder_frame() -> bytes:
    """Generate a 'Camera Connecting or Offline' placeholder JPEG."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Camera Connecting or Offline", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    _, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes()


def _mjpeg_generator():
    """Yield JPEG frames in MJPEG multipart format at ~30 FPS."""
    _placeholder = _placeholder_frame()
    while True:
        with _frame_lock:
            frame_bytes = _latest_frame or _placeholder

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.033)   # ~30 FPS cap


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "name": "SafeVision API",
        "version": "2.0.0",
        "description": "Real-time face recognition security system",
        "endpoints": {
            "/stream": "Live MJPEG video stream",
            "/health": "Health check",
            "/status": "System status & metrics",
            "/faces":  "Recently recognised faces",
        },
    }


@app.get("/health", tags=["monitoring"])
async def health():
    """Health check for load balancers and monitoring."""
    stream_ok = _video_stream is not None and _video_stream.is_alive()
    return JSONResponse(
        status_code=200 if stream_ok else 503,
        content={
            "status": "healthy" if stream_ok else "degraded",
            "stream_connected": stream_ok,
            "uptime_seconds": round(time.time() - _start_time, 1),
        },
    )


@app.get("/status", tags=["monitoring"])
async def status():
    """Detailed system status."""
    face_count = faces_collection.count_documents({})
    reader_alive = _reader_thread is not None and _reader_thread.is_alive()
    detector_alive = _detector_thread is not None and _detector_thread.is_alive()
    recognizer_alive = _recognizer_thread is not None and _recognizer_thread.is_alive()
    return {
        "stream_connected": _video_stream is not None and _video_stream.is_alive(),
        "pipeline": {
            "reader": "running" if reader_alive else "stopped",
            "detector": "running" if detector_alive else "stopped",
            "recognizer": "running" if recognizer_alive else "stopped",
        },
        "fps": _processor.fps if _processor else 0,
        "faces_in_db": face_count,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "config": {
            "frame_size": f"{settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}",
            "detect_every_n": settings.DETECT_EVERY_N,
            "jpeg_quality": settings.STREAM_JPEG_QUALITY,
            "blur_threshold": settings.BLUR_THRESHOLD,
            "yolo_conf": settings.YOLO_CONF_THRESHOLD,
            "recog_threshold": settings.RECOG_THRESHOLD,
            "low_light_enabled": settings.LOW_LIGHT_ENABLE,
        },
    }


@app.get("/stream", tags=["video"])
async def video_stream():
    """
    Live MJPEG video stream with face recognition overlays.

    Open this URL in your mobile app's image/video view component
    to see the annotated camera feed in real time.
    """
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/faces", tags=["recognition"])
async def recent_faces(limit: int = 20):
    """Return the most recently recognised faces."""
    if _processor is None:
        return {"faces": []}
    return {"faces": _processor.get_recent_faces(limit=limit)}


@app.post("/admin/reload-thresholds", tags=["admin"])
async def admin_reload_thresholds():
    """
    Recompute adaptive per-identity recognition thresholds from current
    enrollment data.

    Call this after enrolling new faces with ``scripts/enroll_face.py``
    so the live system picks up the updated calibration without a restart.
    """
    try:
        reload_thresholds()
        return {"status": "ok", "message": "Thresholds reloaded successfully."}
    except Exception as exc:
        logger.error("Failed to reload thresholds: %s", exc)
        return {"status": "error", "message": str(exc)}
