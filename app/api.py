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

# Inter-thread queues
_rtsp_queue: queue.Queue = queue.Queue(maxsize=1)     # raw BGR frames
_detect_queue: queue.Queue = queue.Queue(maxsize=1)   # (preprocessed_frame, boxes)

# Latest raw frame served to WebRTC clients
_latest_raw_frame: np.ndarray | None = None
_frame_lock = threading.Lock()

def get_latest_raw_frame():
    with _frame_lock:
        return _latest_raw_frame

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
    global _latest_raw_frame
    logger.info("Recognizer thread started.")
    while True:
        try:
            frame, boxes = _detect_queue.get(timeout=2.0)
        except queue.Empty:
            if not (_video_stream and _video_stream.is_alive()):
                break
            continue

        result = _processor.recognize_and_annotate(frame, boxes)

        with _frame_lock:
            _latest_raw_frame = result.annotated

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
    await on_shutdown()
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


from aiortc import RTCPeerConnection, RTCSessionDescription
from app.webrtc import SafeVisionTrack

# Global set to keep track of active WebRTC peer connections
_pcs = set()

async def on_shutdown():
    # Close all WebRTC connections gracefully
    coros = [pc.close() for pc in _pcs]
    await asyncio.gather(*coros)
    _pcs.clear()


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

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse

WEBRTC_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeVision Ultra-Low Latency</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body { margin: 0; background-color: #0f172a; color: white; font-family: 'Inter', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; overflow: hidden; }
        .container { background: rgba(30, 41, 59, 0.7); padding: 24px; border-radius: 20px; backdrop-filter: blur(16px); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6); border: 1px solid rgba(255, 255, 255, 0.05); width: 90%; max-width: 1000px; }
        .video-wrapper { position: relative; width: 100%; border-radius: 12px; overflow: hidden; background: #000; box-shadow: 0 4px 20px rgba(0,0,0,0.5); aspect-ratio: 16 / 9; }
        video { width: 100%; height: 100%; object-fit: contain; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
        .title-group { display: flex; align-items: center; gap: 12px; }
        .title { font-size: 1.5rem; font-weight: 600; margin: 0; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .badge { background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.2); padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
        .status { display: flex; align-items: center; gap: 10px; font-size: 0.9rem; color: #94a3b8; font-weight: 600; background: rgba(0,0,0,0.2); padding: 8px 16px; border-radius: 12px; }
        .dot { width: 10px; height: 10px; border-radius: 50%; background-color: #ef4444; transition: all 0.3s ease; box-shadow: 0 0 10px rgba(239, 68, 68, 0.5); }
        .dot.connected { background-color: #22c55e; box-shadow: 0 0 12px #22c55e; }
        .btn { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; border: none; padding: 8px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s; font-family: inherit; }
        .btn:hover { background: linear-gradient(135deg, #60a5fa, #3b82f6); transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }
        .btn:active { transform: translateY(0); }
        .loader { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); display: none; flex-direction: column; align-items: center; gap: 12px; color: #94a3b8; font-weight: 600; }
        .spinner { width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.1); border-top-color: #38bdf8; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-group">
                <h1 class="title">SafeVision WebRTC</h1>
                <span class="badge">Zero Latency</span>
            </div>
            <div class="status">
                <div class="dot" id="status-dot"></div>
                <span id="status-text">Disconnected</span>
                <button class="btn" onclick="start()">Reconnect</button>
            </div>
        </div>
        <div class="video-wrapper">
            <div class="loader" id="loader">
                <div class="spinner"></div>
                <span>Negotiating connection...</span>
            </div>
            <video id="video" autoplay playsinline muted></video>
        </div>
    </div>

    <script>
        let pc = null;

        async function start() {
            const statusText = document.getElementById('status-text');
            const statusDot = document.getElementById('status-dot');
            const loader = document.getElementById('loader');
            const video = document.getElementById('video');
            
            statusText.innerText = 'Connecting...';
            statusDot.classList.remove('connected');
            loader.style.display = 'flex';
            video.style.opacity = '0.5';
            
            if (pc) {
                pc.close();
            }

            pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });
            
            pc.addEventListener('track', (evt) => {
                if (evt.track.kind === 'video') {
                    video.srcObject = evt.streams[0];
                }
            });

            pc.addEventListener('connectionstatechange', () => {
                if (pc.connectionState === 'connected') {
                    statusText.innerText = 'Live';
                    statusDot.classList.add('connected');
                    loader.style.display = 'none';
                    video.style.opacity = '1';
                } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
                    statusText.innerText = 'Disconnected';
                    statusDot.classList.remove('connected');
                    loader.style.display = 'none';
                }
            });

            // We only want to receive video
            pc.addTransceiver('video', { direction: 'recvonly' });

            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Wait a moment for ICE candidates to gather (trickle ICE is better, but this is simpler)
            await new Promise(resolve => setTimeout(resolve, 500));

            try {
                const response = await fetch('/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });

                if (!response.ok) throw new Error('Failed to get answer');
                
                const answer = await response.json();
                await pc.setRemoteDescription(answer);
            } catch (err) {
                console.error('Signaling error:', err);
                statusText.innerText = 'Connection Error';
                statusDot.classList.remove('connected');
                loader.style.display = 'none';
            }
        }
        
        // Auto-start connection on page load
        window.addEventListener('load', start);
    </script>
</body>
</html>
"""

@app.get("/player", tags=["video"])
async def webrtc_player():
    """Returns a beautiful HTML page that automatically connects to the WebRTC stream."""
    return HTMLResponse(content=WEBRTC_HTML)

@app.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "name": "SafeVision API",
        "version": "2.0.0",
        "description": "Real-time face recognition security system",
        "endpoints": {
            "/stream": "Live MJPEG video stream",
            "/player": "WebRTC Video Player (Browser interface)",
            "/health": "Health check",
            "/status": "System status & metrics",
            "/faces":  "Recently recognised faces",
            "/offer":  "WebRTC offer endpoint",
        },
    }


def _generate_mjpeg():
    while True:
        frame = get_latest_raw_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        
        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), settings.STREAM_JPEG_QUALITY])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        # Yield to allow other tasks (approx matching the camera FPS)
        time.sleep(1 / 30.0)

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


from pydantic import BaseModel
import asyncio

class WebRTCOffer(BaseModel):
    sdp: str
    type: str

@app.post("/offer", tags=["video"])
async def webrtc_offer(offer: WebRTCOffer):
    """
    WebRTC endpoint for ultra-low latency video streaming.
    
    Accepts an SDP offer and returns an SDP answer.
    """
    pc = RTCPeerConnection()
    _pcs.add(pc)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            _pcs.discard(pc)

    # Attach our custom SafeVision video track
    pc.addTrack(SafeVisionTrack())

    # Set the remote description
    offer_sdp = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    await pc.setRemoteDescription(offer_sdp)
    
    # Create and set the local description
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/faces", tags=["recognition"], dependencies=[Depends(verify_token)])
async def recent_faces(limit: int = 20):
    """Return the most recently recognised faces."""
    if _processor is None:
        return {"faces": []}
    return {"faces": _processor.get_recent_faces(limit=limit)}


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
