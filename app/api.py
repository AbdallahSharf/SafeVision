"""
FastAPI application for SafeVision.

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
import threading
import time
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.database import faces_collection
from app.stream import VideoStream
from app.processor import FrameProcessor

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Shared state (populated on startup)
# ---------------------------------------------------------------------------
_video_stream: VideoStream | None = None
_processor: FrameProcessor | None = None
_bg_thread: threading.Thread | None = None
_latest_frame: bytes = b""
_frame_lock = threading.Lock()
_start_time: float = 0.0


# ---------------------------------------------------------------------------
# Background processing loop
# ---------------------------------------------------------------------------
def _processing_loop() -> None:
    """
    Continuously read frames from the RTSP stream, process them through the
    full SafeVision pipeline, and store the latest JPEG-encoded result for
    the MJPEG endpoint.
    """
    global _latest_frame

    logger.info("Background processing loop started.")
    while _video_stream and _video_stream.is_alive():
        frame = _video_stream.read()
        if frame is None:
            continue

        result = _processor.process(frame)

        # Encode the annotated frame as JPEG for streaming
        _, jpeg = cv2.imencode(".jpg", result.annotated, [cv2.IMWRITE_JPEG_QUALITY, settings.STREAM_JPEG_QUALITY])
        with _frame_lock:
            _latest_frame = jpeg.tobytes()

    logger.info("Background processing loop exited.")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the stream + processor on startup; clean up on shutdown."""
    global _video_stream, _processor, _bg_thread, _start_time

    _start_time = time.time()

    logger.info("Starting VideoStream …")
    _video_stream = VideoStream()
    _processor = FrameProcessor()

    _bg_thread = threading.Thread(target=_processing_loop, daemon=True)
    _bg_thread.start()

    logger.info("SafeVision API is ready.")
    yield  # ── app is running ──

    # Shutdown
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
    version="1.0.0",
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
def _mjpeg_generator():
    """Yield JPEG frames in MJPEG multipart format."""
    while True:
        with _frame_lock:
            frame_bytes = _latest_frame

        if not frame_bytes:
            import numpy as np
            # Create a black placeholder image with text
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Camera Connecting or Offline", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, jpeg = cv2.imencode(".jpg", img)
            frame_bytes = jpeg.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        # ~30 FPS cap to avoid overwhelming the client
        time.sleep(0.033)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "name": "SafeVision API",
        "version": "1.0.0",
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
    return {
        "stream_connected": _video_stream is not None and _video_stream.is_alive(),
        "fps": _processor.fps if _processor else 0,
        "faces_in_db": face_count,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "config": {
            "frame_size": f"{settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}",
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
