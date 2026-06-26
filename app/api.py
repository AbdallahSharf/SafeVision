"""
FastAPI application for SafeVision.

Phase 4 refactor: performance optimizations.

  - Event-driven MJPEG streaming (replaces CPU-burning spin-loop)
  - FAISS local index initialization and periodic sync
  - All existing endpoints preserved

Pipeline Architecture (4 threads + inference/DB background threads):

  Thread 1 — _reader_loop()
      Reads raw frames from the RTSP VideoStream and puts them into
      _rtsp_queue. Ensures the network buffer never overflows.

  Thread 2 — _detector_loop()
      Pops raw frames from _rtsp_queue, runs YOLO detection, and pushes 
      (frame, boxes) pairs into _detect_queue.

  Thread 3 — _recognizer_loop()
      Pops (frame, boxes) from _detect_queue, runs ArcFace, and stores 
      the result in _latest_ai_results.

  Thread 4 — _jpeg_encoder_loop()
      Takes the freshest camera frame and the latest AI boxes, draws the
      overlays, encodes to JPEG, and updates the MJPEG streamer state.

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

import os

import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.database import faces_collection
from app.stream import VideoStream
from app.processor import FrameProcessor
from app.recognition import reload_thresholds, load_faiss_index, get_faiss_index
from app.alerts import init_firebase

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Lock-free shared state — separate atomic variables (Component 4)
# Each thread only writes its own variable, no read-modify-write races.
# Python GIL makes single reference assignment atomic.
# ---------------------------------------------------------------------------
_video_stream: VideoStream | None = None
_processor: FrameProcessor | None = None
_start_time: float = 0.0

# Queues for inter-thread communication
_rtsp_queue: queue.Queue = queue.Queue(maxsize=1)
_detect_queue: queue.Queue = queue.Queue(maxsize=1)

# Atomic variables — each written by exactly ONE thread (no races)
_latest_raw_frame: np.ndarray | None = None         # written by reader thread only
_latest_ai_results: list = []                       # written by recognizer thread only
_latest_ai_fps: float = 0.0                         # written by recognizer thread only

# Pre-encoded JPEG bytes from the encoder thread
_latest_jpeg_bytes: bytes | None = None
_latest_frame_id: int = 0

# Event fired by the reader every time a new frame arrives.
# The JPEG encoder and MJPEG generator wait on this instead of spinning.
_new_frame_event: threading.Event = threading.Event()

# Event fired by the reader to notify the YOLO detector thread
_new_detector_frame_event: threading.Event = threading.Event()

# Thread handles
_reader_thread: threading.Thread | None = None
_detector_thread: threading.Thread | None = None
_recognizer_thread: threading.Thread | None = None
_jpeg_encoder_thread: threading.Thread | None = None
_faiss_sync_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Main Pipeline Thread
# ---------------------------------------------------------------------------
def _reader_loop() -> None:
    global _latest_raw_frame
    logger.info("Reader thread started.")
    while _video_stream and _video_stream.is_alive():
        # Block until the VideoStream reader thread signals a new frame is ready
        _video_stream.frame_ready_event.wait(timeout=1.0)
        _video_stream.frame_ready_event.clear()

        frame = _video_stream.read_latest()
        if frame is not None:
            _latest_raw_frame = frame
            # Signal the encoder and detector threads that a new frame is ready
            _new_frame_event.set()
            _new_detector_frame_event.set()

def _detector_loop() -> None:
    logger.info("Detector thread started.")
    last_detected_frame = None
    last_detect_time = 0.0
    # Run YOLO at max 5 FPS. YOLO holds the GPU lock for ~100ms per call.
    # Without this limit, YOLO runs on every camera frame (25 FPS) and holds
    # the GPU lock 100% of the time, starving ArcFace and causing 1 FPS AI output.
    # At 5 FPS detection, YOLO uses only ~50% of GPU lock time, leaving the rest
    # for ArcFace to run smoothly.
    min_detect_interval = 1.0 / 5  # 200ms between YOLO calls
    while _video_stream and _video_stream.is_alive():
        try:
            # Block until reader thread signals a new frame is available
            _new_detector_frame_event.wait(timeout=1.0)
            _new_detector_frame_event.clear()

            frame = _latest_raw_frame
            if frame is None or frame is last_detected_frame:
                continue

            # Rate-limit: skip frames if it's too soon since last YOLO call
            now = time.time()
            if now - last_detect_time < min_detect_interval:
                continue

            last_detected_frame = frame
            last_detect_time = now
            t0 = time.time()
            raw_frame, boxes = _processor.detect(frame)
            t1 = time.time()
            if t1 - t0 > 0.05:
                logger.warning(f"Detect took {t1-t0:.3f}s")
            # Drop stale detections if recognizer is busy; always keep newest
            try:
                _detect_queue.get_nowait()
            except queue.Empty:
                pass
            _detect_queue.put_nowait((raw_frame, boxes))
        except Exception as exc:
            logger.error("Detector thread error: %s", exc)

def _recognizer_loop() -> None:
    global _latest_ai_results, _latest_ai_fps
    logger.info("Recognizer thread started.")
    while _video_stream and _video_stream.is_alive():
        try:
            frame, boxes = _detect_queue.get(timeout=1.0)
            t0 = time.time()
            ai_result = _processor.recognize_faces(frame, boxes)
            t1 = time.time()
            if t1 - t0 > 0.05:
                logger.warning(f"Recognize took {t1-t0:.3f}s for {len(boxes)} boxes")
            _latest_ai_results = ai_result.faces
            _latest_ai_fps = ai_result.fps
        except queue.Empty:
            continue
        except Exception as exc:
            logger.error("Recognizer thread error: %s", exc)

def _jpeg_encoder_loop() -> None:
    global _latest_jpeg_bytes, _latest_frame_id
    logger.info("JPEG encoder thread started.")
    min_interval = 1.0 / max(1, settings.TARGET_STREAM_FPS)
    last_encode_time = 0.0

    while _video_stream and _video_stream.is_alive():
        # Block here until the reader signals a new frame — no CPU burn
        _new_frame_event.wait(timeout=1.0)
        _new_frame_event.clear()

        now = time.time()
        if now - last_encode_time < min_interval:
            # Cap encoding rate to TARGET_STREAM_FPS, but don't sleep —
            # just skip this frame and wait for the next event.
            continue
        last_encode_time = now

        frame = _latest_raw_frame
        if frame is None:
            continue

        # frame is already pre-scaled to FRAME_WIDTH x FRAME_HEIGHT by stream.py
        serve = frame.copy()

        # Draw bounding boxes from _latest_ai_results
        h, w = serve.shape[:2]
        sx_scale = w / 1280.0
        sy_scale = h / 720.0
        base_thickness = max(1, int(2 * min(sx_scale, sy_scale)))
        font_scale = max(0.5, 0.8 * min(sx_scale, sy_scale))

        faces = list(_latest_ai_results)
        fps = _latest_ai_fps

        for face in faces:
            bx1, by1, bx2, by2 = face.bbox
            color = (0, 255, 0) if face.name not in ("Unauthorized", "Unknown", "Too Blurry") else (0, 0, 255)
            cv2.rectangle(serve, (bx1, by1), (bx2, by2), color, base_thickness)

            label = face.name
            cv2.putText(
                serve, label, (bx1, max(10, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, base_thickness
            )

        fps_scale = max(0.6, 0.8 * min(sx_scale, sy_scale))
        fps_thickness = max(1, int(2 * min(sx_scale, sy_scale)))
        cv2.putText(
            serve, f"AI FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, fps_scale, (255, 255, 0), fps_thickness
        )

        _, jpeg = cv2.imencode(
            ".jpg", serve,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.STREAM_JPEG_QUALITY]
        )

        _latest_jpeg_bytes = jpeg.tobytes()
        _latest_frame_id += 1


# ---------------------------------------------------------------------------
# FAISS periodic sync thread
# ---------------------------------------------------------------------------
def _faiss_sync_loop() -> None:
    """
    Periodically re-sync the local FAISS index with MongoDB to pick up
    new enrollments made externally (e.g., via enroll_face.py).
    """
    interval = settings.FAISS_SYNC_INTERVAL
    logger.info("FAISS sync thread started (interval=%ds).", interval)
    while True:
        time.sleep(interval)
        try:
            faiss_index = get_faiss_index()
            count = faiss_index.rebuild(faces_collection)
            # Also reload adaptive thresholds since enrollments may have changed
            reload_thresholds()
            if _processor is not None:
                _processor.clear_cache()
            logger.info("FAISS periodic sync complete — %d embeddings.", count)
            
            # Clean up unauthorized faces older than 7 days
            try:
                faces_dir = os.environ.get(
                    "UNAUTHORIZED_FACES_DIR",
                    "/opt/safevision/unauthorized_faces" if os.name != "nt" else os.path.join(os.getcwd(), "unauthorized_faces"),
                )
                if os.path.exists(faces_dir):
                    now_time = time.time()
                    deleted_count = 0
                    for filename in os.listdir(faces_dir):
                        file_path = os.path.join(faces_dir, filename)
                        if os.path.isfile(file_path):
                            # 7 days = 7 * 24 * 60 * 60 = 604800 seconds
                            if os.stat(file_path).st_mtime < now_time - 604800:
                                os.remove(file_path)
                                deleted_count += 1
                    if deleted_count > 0:
                        logger.info("Cleaned up %d old unauthorized face photos (older than 7 days).", deleted_count)
            except Exception as e:
                logger.warning("Failed to clean up old faces: %s", e)

        except Exception as exc:
            logger.warning("FAISS periodic sync failed: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background pipeline on startup; clean up on shutdown."""
    global _video_stream, _processor, _start_time
    global _reader_thread, _detector_thread, _recognizer_thread, _jpeg_encoder_thread, _faiss_sync_thread

    _start_time = time.time()

    logger.info("Starting VideoStream …")
    _video_stream = VideoStream()
    _processor = FrameProcessor()

    # Initialize Firebase for push notifications
    init_firebase()

    # Load FAISS index from MongoDB for instant local matching
    try:
        load_faiss_index()
    except Exception as exc:
        logger.warning("FAISS index failed to load — will use MongoDB fallback: %s", exc)

    _reader_thread = threading.Thread(target=_reader_loop, daemon=True, name="ReaderThread")
    _reader_thread.start()

    _detector_thread = threading.Thread(target=_detector_loop, daemon=True, name="DetectorThread")
    _detector_thread.start()
    
    _recognizer_thread = threading.Thread(target=_recognizer_loop, daemon=True, name="RecognizerThread")
    _recognizer_thread.start()

    _jpeg_encoder_thread = threading.Thread(target=_jpeg_encoder_loop, daemon=True, name="JPEGEncoderThread")
    _jpeg_encoder_thread.start()

    # Start FAISS periodic sync thread
    _faiss_sync_thread = threading.Thread(target=_faiss_sync_loop, daemon=True, name="sv-faiss-sync")
    _faiss_sync_thread.start()

    logger.info("SafeVision API is ready — multi-threaded pipeline active.")
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
    version="3.0.0",
    lifespan=lifespan,
)

# Allow mobile apps to connect from any origin
# Note: allow_credentials is disabled with wildcard origins for security.
# For production, replace "*" with your specific frontend domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
        "version": "3.0.0",
        "description": "Real-time face recognition security system",
        "endpoints": {
            "/stream": "Live MJPEG video stream",
            "/health": "Health check",
            "/status": "System status & metrics",
            "/faces":  "Recently recognised faces",
        },
    }


# ---------------------------------------------------------------------------
# MJPEG Stream Endpoint
# ---------------------------------------------------------------------------
def _generate_mjpeg():
    """
    Low-latency MJPEG generator — yields pre-encoded JPEG bytes.

    Waits on _new_frame_event instead of spinning with time.sleep().
    This avoids the 10ms polling anti-pattern that can starve other
    async FastAPI coroutines.
    """
    last_yielded_id = -1

    while True:
        # Block until the encoder signals a new JPEG is ready
        _new_frame_event.wait(timeout=1.0)

        current_id = _latest_frame_id
        if current_id == last_yielded_id:
            continue

        last_yielded_id = current_id
        jpeg_bytes = _latest_jpeg_bytes
        if jpeg_bytes is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
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
async def system_status():
    """Detailed system status."""
    face_count = faces_collection.count_documents({})
    
    threads_alive = {
        "reader": _reader_thread is not None and _reader_thread.is_alive(),
        "detector": _detector_thread is not None and _detector_thread.is_alive(),
        "recognizer": _recognizer_thread is not None and _recognizer_thread.is_alive(),
        "encoder": _jpeg_encoder_thread is not None and _jpeg_encoder_thread.is_alive(),
    }
    
    faiss_index = get_faiss_index()
    return {
        "stream_connected": _video_stream is not None and _video_stream.is_alive(),
        "pipeline": threads_alive,
        "fps": _processor.fps if _processor else 0,
        "faces_in_db": face_count,
        "faiss_index": {
            "loaded": faiss_index.is_loaded,
            "embeddings": faiss_index.total_embeddings,
        },
        "uptime_seconds": round(time.time() - _start_time, 1),
        "config": {
            "frame_size": f"{settings.FRAME_WIDTH}x{settings.FRAME_HEIGHT}",
            "jpeg_quality": settings.STREAM_JPEG_QUALITY,
            "target_stream_fps": settings.TARGET_STREAM_FPS,
            "detect_every_n_frames": settings.DETECT_EVERY_N_FRAMES,
            "blur_threshold": settings.BLUR_THRESHOLD,
            "yolo_conf": settings.YOLO_CONF_THRESHOLD,
            "recog_threshold": settings.RECOG_THRESHOLD,
            "low_light_enabled": settings.LOW_LIGHT_ENABLE,
            "gpu_decode": settings.USE_GPU_DECODE,
            "faiss_sync_interval": settings.FAISS_SYNC_INTERVAL,
            "alert_cooldown": settings.ALERT_COOLDOWN_SECONDS,
        },
    }




@app.get("/faces", tags=["recognition"], dependencies=[Depends(verify_token)])
async def recent_faces(limit: int = 20):
    """Return the most recently recognised faces."""
    limit = max(1, min(limit, 100))  # clamp to [1, 100]
    if _processor is None:
        return {"faces": []}
    return {"faces": _processor.get_recent_faces(limit=limit)}


from fastapi.staticfiles import StaticFiles
import os
from app.database import async_alerts_collection

# Create image directory if it doesn't exist
_UNAUTHORIZED_FACES_DIR = os.environ.get(
    "UNAUTHORIZED_FACES_DIR",
    os.path.join(os.getcwd(), "unauthorized_faces")
    if os.name == "nt"
    else "/opt/safevision/unauthorized_faces",
)
os.makedirs(_UNAUTHORIZED_FACES_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=_UNAUTHORIZED_FACES_DIR), name="images")

@app.get("/alerts", tags=["monitoring"], dependencies=[Depends(verify_token)])
async def get_alerts(limit: int = 20):
    """Return the most recent unauthorized access alerts for the mobile app history view."""
    limit = max(1, min(limit, 100))  # clamp to [1, 100]
    cursor = async_alerts_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    alerts = await cursor.to_list(length=limit)
    return {"alerts": alerts}

@app.post("/admin/reload-thresholds", tags=["admin"], dependencies=[Depends(verify_token)])
async def admin_reload_thresholds():
    """
    Recompute adaptive per-identity recognition thresholds from current
    enrollment data and rebuild the FAISS index.

    Call this after enrolling new faces with ``scripts/enroll_face.py``
    so the live system picks up the updated calibration without a restart.
    """
    try:
        reload_thresholds()
        count = get_faiss_index().rebuild(faces_collection)
        if _processor is not None:
            _processor._track_cache.clear()
        return {
            "status": "ok",
            "message": "Thresholds reloaded, FAISS index rebuilt, and track cache cleared.",
            "faiss_embeddings": count,
        }
    except Exception as exc:
        logger.error("Failed to reload thresholds: %s", exc)
        return {"status": "error", "message": str(exc)}
