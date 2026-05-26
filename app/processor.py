"""
Frame processing pipeline for SafeVision.

Phase 2 refactor: the single process() method is now split into two stages
so they can run in separate threads (see api.py):

  Stage 1 — detect(frame)
      Runs YOLO every DETECT_EVERY_N frames (frame skipping from Phase 1).
      Returns the raw frame + list of bounding boxes.

  Stage 2 — recognize_and_annotate(frame, boxes)
      Runs the blur gate, ArcFace embedding, MongoDB vector search, and
      ByteTrack-based temporal smoothing.  Draws bounding boxes and labels.
      Returns a ProcessedFrame ready for MJPEG encoding.
"""

from app.config import settings
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from app.enhancement import enhance_frame, enhance_face
from app.models_loader import get_yolo, get_arcface, _DEVICE
import asyncio
from app.recognition import recognize_face, async_recognize_face
from app.tracker import FaceTracker
from app.alerts import send_unauthorized_alert

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Async DB Background Loop
# ---------------------------------------------------------------------------
_db_loop = asyncio.new_event_loop()
def _run_db_loop():
    asyncio.set_event_loop(_db_loop)
    _db_loop.run_forever()

_db_thread = threading.Thread(target=_run_db_loop, daemon=True, name="sv-db-loop")
_db_thread.start()

import queue
_inference_queue = queue.Queue(maxsize=100)

def _run_inference_loop():
    """
    Background inference thread.

    Drains up to BATCH_SIZE items from the queue per iteration so that
    multiple faces visible simultaneously are processed in a single batched
    ONNX call (``get_feats``).  On GPU this takes nearly the same wall-clock
    time as a single face, giving a proportional throughput improvement.
    """
    BATCH_SIZE = 4
    while True:
        try:
            # Block on the first item, then greedily drain more without waiting
            items = [_inference_queue.get()]
            if items[0] is None:
                continue
            while len(items) < BATCH_SIZE:
                try:
                    extra = _inference_queue.get_nowait()
                    if extra is not None:
                        items.append(extra)
                except queue.Empty:
                    break

            arcface = get_arcface()
            faces_rgb = [it[0] for it in items]

            # Single batched ONNX inference call — much cheaper than N individual calls
            embeddings = arcface.get_feats(faces_rgb)  # shape (N, 512)

            for (face_rgb, face_bgr, track, box), embedding in zip(items, embeddings):
                embedding = embedding.flatten()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                    # 2. Async DB lookup
                    async def _do_recognize(emb, trk, bx, raw_face):
                        try:
                            ident, scr = await async_recognize_face(emb)
                            trk.identity_history.append(ident)
                            trk.last_score = scr
                            if ident == "Unauthorized":
                                send_unauthorized_alert(scr, bx, raw_face)
                        finally:
                            trk.is_recognizing = False

                    asyncio.run_coroutine_threadsafe(
                        _do_recognize(embedding, track, box, face_bgr),
                        _db_loop
                    )
                else:
                    track.is_recognizing = False

        except Exception as exc:
            logger.error("Inference thread error: %s", exc)
            # Best-effort: release any locks held by items in this batch
            for it in locals().get("items", []):
                try:
                    if hasattr(it[2], 'is_recognizing'):
                        it[2].is_recognizing = False
                except Exception:
                    pass

_inference_thread = threading.Thread(target=_run_inference_loop, daemon=True, name="sv-inference")
_inference_thread.start()

# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------
@dataclass
class DetectedFace:
    """Single recognised face in a frame."""

    name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessedFrame:
    """Result of processing a single video frame."""

    annotated: np.ndarray  # BGR frame with bounding boxes drawn
    faces: List[DetectedFace] = field(default_factory=list)
    fps: float = 0.0


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------
class FrameProcessor:
    """
    Stateful processor that runs the full SafeVision pipeline on each frame.

    The pipeline is split into two stages for the async 3-thread architecture:

      1. detect(frame) → (preprocessed_frame, boxes)      [YOLO, runs in Thread 2]
      2. recognize_and_annotate(frame, boxes) → ProcessedFrame  [ArcFace, Thread 3]
    """

    def __init__(self):
        # ByteTrack-style IoU tracker — gives each face a persistent ID
        self._tracker = FaceTracker(
            iou_threshold=0.3,
            max_lost_frames=8,
        )

        # Frame skipping — YOLO runs every DETECT_EVERY_N frames
        self._frame_idx: int = 0
        self._cached_boxes: List[Tuple] = []   # boxes from last YOLO run

        # FPS tracking
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps = 0.0

        # Recent faces buffer (for the /faces REST endpoint)
        self._recent_faces: deque[DetectedFace] = deque(maxlen=50)
        self._lock = threading.Lock()

    # ── Stage 1: Detection ────────────────────────────────────────────────
    def detect(self, raw_frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Stage 1 — preprocess + YOLO detection (runs in the detector thread).

        Runs YOLO on every frame to ensure ByteTrack gets accurate positions
        for smooth tracking.

        Parameters
        ----------
        raw_frame : np.ndarray
            Raw BGR frame from the RTSP reader.

        Returns
        -------
        (preprocessed_frame, boxes)
            boxes is a list of (x1, y1, x2, y2) tuples — one per detected face.
        """
        frame = cv2.resize(raw_frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
        frame = enhance_frame(frame)

        self._frame_idx += 1
        if self._frame_idx % settings.DETECT_EVERY_N != 0:
            return frame, self._cached_boxes

        yolo = get_yolo()
        results = yolo(
            frame,
            conf=settings.YOLO_CONF_THRESHOLD,
            imgsz=settings.IMGSZ,
            device=_DEVICE,
            verbose=False,
        )
        h, w = frame.shape[:2]
        boxes = []
        for r in results:
            for i in range(len(r.boxes)):
                if float(r.boxes.conf[i]) < settings.BOX_CONF_THRESHOLD:
                    continue
                cls = int(r.boxes.cls[i])
                if yolo.names.get(cls, "").lower() != "face":
                    continue
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                # Expand by margin
                x1 = max(0, x1 - settings.FACE_MARGIN)
                y1 = max(0, y1 - settings.FACE_MARGIN)
                x2 = min(w, x2 + settings.FACE_MARGIN)
                y2 = min(h, y2 + settings.FACE_MARGIN)
                boxes.append((x1, y1, x2, y2))

        self._cached_boxes = boxes
        return frame, boxes

    # ── Stage 2: Recognition + Annotation ────────────────────────────────
    def recognize_and_annotate(
        self,
        frame: np.ndarray,
        boxes: List[Tuple],
    ) -> "ProcessedFrame":
        """
        Stage 2 — ArcFace recognition + ByteTrack smoothing + annotation.

        Runs in the recognizer thread, consuming (frame, boxes) pairs
        produced by the detector thread.
        """
        # No longer lazy-loading arcface here because it's done in the background thread
        detected_faces: List[DetectedFace] = []

        # Update tracker with current detections
        active_tracks = self._tracker.update(boxes)

        # Build a dict of track → identity for all tracks that have history
        smoothed: dict[int, str] = {}
        for track in active_tracks:
            if track.smoothed_identity:
                smoothed[track.track_id] = track.smoothed_identity

        # Identify which tracks need to be recognized this frame
        for track in active_tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # ── Blur quality gate — cached per track ────────────────────────
            # Recompute only every 5 frames: blur changes slowly, and
            # cv2.Laplacian on a crop costs 3–8 ms each call.
            _blur_stale = (
                not hasattr(track, '_blur_cache_frame')
                or (self._frame_count - track._blur_cache_frame) >= 5
            )
            if _blur_stale:
                blur_score = cv2.Laplacian(face, cv2.CV_64F).var()
                track._blur_score = blur_score
                track._blur_cache_frame = self._frame_count
            else:
                blur_score = track._blur_score
            if blur_score < settings.BLUR_THRESHOLD:
                # Still annotate with last known identity if available
                identity = smoothed.get(track.track_id, "Unknown")
                score = 0.0
            else:
                if not hasattr(track, 'is_recognizing'):
                    track.is_recognizing = False

                # Only run heavy ArcFace if this track has no identity yet,
                # or periodically (staggered by track_id) to verify they haven't swapped.
                # Crucially, skip if we are ALREADY querying the DB for this track!
                needs_recognition = not track.is_recognizing and (not track.identity_history or self._frame_count % 15 == track.track_id % 15)
                
                if needs_recognition:
                    track.is_recognizing = True  # Lock this track to prevent spamming the DB
                    
                    # Pre-process
                    face_enhanced = enhance_face(face)
                    face_resized = cv2.resize(face_enhanced, (settings.FACE_SIZE, settings.FACE_SIZE))
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    
                    try:
                        _inference_queue.put_nowait((face_rgb, face.copy(), track, (x1, y1, x2, y2)))
                    except queue.Full:
                        logger.warning("Inference queue full — dropping face")
                        track.is_recognizing = False
                
                # Always define identity (use smoothed identity from tracker)
                identity = track.smoothed_identity or "Checking..."
                score = track.last_score

            # ── Annotate ───────────────────────────────────────────────────
            color = (0, 255, 0) if identity not in ("Unauthorized", "Unknown", "Checking...") else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{identity} ({score:.2f})" if score > 0 else identity
            cv2.putText(
                frame,
                label,
                (x1, max(10, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            
            # Track ID badge (small, top-right of box)
            cv2.putText(
                frame,
                f"#{track.track_id}",
                (x2 - 30, max(15, y1 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )

            detected_faces.append(DetectedFace(
                name=identity,
                confidence=round(score, 4),
                bbox=(x1, y1, x2, y2),
            ))

        # Update recent faces buffer
        with self._lock:
            self._recent_faces.extend(detected_faces)

        # FPS calculation
        self._frame_count += 1
        if self._frame_count >= 10:
            now = time.time()
            self._fps = self._frame_count / (now - self._fps_time)
            self._fps_time = now
            self._frame_count = 0

        # Draw FPS overlay
        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

        return ProcessedFrame(
            annotated=frame,
            faces=detected_faces,
            fps=round(self._fps, 2),
        )

    # ── Convenience wrapper (used by tests / local dev) ──────────────────
    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Run both stages sequentially (single-threaded path)."""
        preprocessed, boxes = self.detect(frame)
        return self.recognize_and_annotate(preprocessed, boxes)

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def fps(self) -> float:
        return round(self._fps, 2)

    def get_recent_faces(self, limit: int = 20) -> List[dict]:
        """Return the most recently detected faces as JSON-serialisable dicts."""
        with self._lock:
            faces = list(self._recent_faces)[-limit:]
        return [
            {
                "name": f.name,
                "confidence": f.confidence,
                "bbox": list(f.bbox),
                "timestamp": f.timestamp,
            }
            for f in reversed(faces)
        ]
