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
      FrameProcessor (ArcFace + tracker), and stores the result
      in _latest_ai_results.

The MJPEG generator runs independently, pulling the freshest raw frame
from Thread 1 and the latest AI results from Thread 3, guaranteeing
zero-latency video.

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
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.database import faces_collection
from app.stream import VideoStream
from app.processor import FrameProcessor
from app.recognition import reload_thresholds
from app.alerts import init_firebase

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Shared state (populated on startup)
# ---------------------------------------------------------------------------
_video_stream: VideoStream | None = None
_processor: FrameProcessor | None = None
_start_time: float = 0.0

# Inter-thread queues — size 2 gives a one-frame burst cushion without accumulating lag
_rtsp_queue: queue.Queue = queue.Queue(maxsize=2)     # raw BGR frames
_detect_queue: queue.Queue = queue.Queue(maxsize=2)   # (preprocessed_frame, boxes)

# Latest AI results (faces, names, boxes)
_latest_ai_results: list = []
# Latest RAW frame direct from camera (fast path)
_latest_raw_unprocessed: np.ndarray | None = None
_frame_lock = threading.Lock()

def get_latest_fast_frame():
    """Return the latest raw camera frame for the low-latency MJPEG fast-path."""
    with _frame_lock:
        return _latest_raw_unprocessed

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

    Uses read_latest() to always grab the single freshest frame from the
    VideoStream, discarding any intermediate frames that built up while the
    detector was busy.  Also writes every raw frame directly to
    _latest_raw_unprocessed for the low-latency MJPEG fast-path.
    """
    global _latest_raw_unprocessed
    logger.info("Reader thread started.")
    while _video_stream and _video_stream.is_alive():
        # read_latest() drains the queue and returns only the freshest frame
        frame = _video_stream.read_latest()
        if frame is None:
            # Queue was empty — wait a short time then try again
            time.sleep(0.005)
            continue

        # Fast-path: expose raw frame immediately for MJPEG (no pipeline delay)
        with _frame_lock:
            _latest_raw_unprocessed = frame

        # Push into YOLO/recognition pipeline (drops oldest if detector is behind)
        try:
            _rtsp_queue.put_nowait(frame)
        except queue.Full:
            try:
                _rtsp_queue.get_nowait()
            except queue.Empty:
                pass
            _rtsp_queue.put_nowait(frame)
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
    Thread 3 — ArcFace recognizer.

    Pops (frame, boxes) from _detect_queue, runs Stage 2 of
    FrameProcessor (ArcFace + ByteTrack smoothing), and stores
    the result in _latest_ai_results.
    """
    global _latest_ai_results
    logger.info("Recognizer thread started.")
    while True:
        try:
            frame, boxes = _detect_queue.get(timeout=2.0)
        except queue.Empty:
            if not (_video_stream and _video_stream.is_alive()):
                break
            continue

        try:
            result = _processor.recognize_faces(frame, boxes)

            with _frame_lock:
                _latest_ai_results = result.faces
        except Exception as exc:
            logger.error("Recognizer thread error: %s", exc)

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

    # Initialize Firebase for push notifications
    init_firebase()

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
# Authentication
# ---------------------------------------------------------------------------
security = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    """Validate Bearer token for protected routes."""
    if not settings.API_SECRET_KEY:
        # If no secret is configured, allow all (useful for dev)
        return
        
    if not credentials or credentials.credentials != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing authentication token",
        )

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


def _generate_mjpeg():
    """
    Low-latency MJPEG generator.

    Architecture: Decoupled AI streaming
    ────────────────────────────
    The generator pulls the absolute freshest raw frame from the camera,
    scales it, draws the latest known bounding boxes on it, and sends it.
    This runs completely asynchronously to the AI pipeline, guaranteeing
    a smooth stream (e.g. 30 FPS) even if the AI is processing at 15 FPS.
    """
    last_frame_id: int | None = None
    while True:
        with _frame_lock:
            raw = _latest_raw_unprocessed
            ai_results = list(_latest_ai_results)  # shallow copy for thread safety
            current_fps = _processor.fps if _processor else 0.0

        if raw is None:
            time.sleep(0.01)
            continue

        frame_id = id(raw)
        if frame_id == last_frame_id:
            time.sleep(0.004)   # ~4 ms spin — keeps CPU low while waiting for a new frame
            continue
        last_frame_id = frame_id

        # Resize to match the AI bounding box coordinate space
        serve = cv2.resize(raw, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
        
        # Draw the latest AI results on the frame
        for face in ai_results:
            x1, y1, x2, y2 = face.bbox
            identity = face.name
            score = face.confidence
            
            color = (0, 255, 0) if identity not in ("Unauthorized", "Unknown", "Checking...") else (0, 0, 255)
            cv2.rectangle(serve, (x1, y1), (x2, y2), color, 2)
            
            label = f"{identity} ({score:.2f})" if score > 0 else identity
            cv2.putText(
                serve, label, (x1, max(10, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        # Draw AI FPS overlay
        cv2.putText(
            serve, f"AI FPS: {current_fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
        )

        _, jpeg = cv2.imencode(
            ".jpg", serve,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.STREAM_JPEG_QUALITY]
        )
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )

@app.get("/stream", tags=["video"])
async def stream():
    """Live MJPEG video stream for legacy clients and browsers."""
    if _processor is None:
        raise HTTPException(status_code=503, detail="System starting up")
        
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive"
    }
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers
    )


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




@app.get("/faces", tags=["recognition"], dependencies=[Depends(verify_token)])
async def recent_faces(limit: int = 20):
    """Return the most recently recognised faces."""
    if _processor is None:
        return {"faces": []}
    return {"faces": _processor.get_recent_faces(limit=limit)}


from fastapi.staticfiles import StaticFiles
import os
from app.database import async_alerts_collection

# Create image directory if it doesn't exist
os.makedirs("/opt/safevision/unauthorized_faces", exist_ok=True)
app.mount("/images", StaticFiles(directory="/opt/safevision/unauthorized_faces"), name="images")

@app.get("/alerts", tags=["monitoring"], dependencies=[Depends(verify_token)])
async def get_alerts(limit: int = 20):
    """Return the most recent unauthorized access alerts for the mobile app history view."""
    cursor = async_alerts_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    alerts = await cursor.to_list(length=limit)
    return {"alerts": alerts}

@app.post("/admin/reload-thresholds", tags=["admin"], dependencies=[Depends(verify_token)])
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
