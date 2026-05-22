"""
Frame processing pipeline for SafeVision.

Encapsulates the full detect → recognise → annotate cycle so that both
the MJPEG streaming endpoint and any future consumers share the same logic.
"""

from app.config import settings
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from app.enhancement import enhance_frame, enhance_face
from app.models_loader import get_yolo, get_arcface, _DEVICE
from app.recognition import recognize_face

logger = logging.getLogger("safevision")


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

    Maintains a short history for temporal smoothing of identity labels.
    """

    def __init__(self):
        # Per-face temporal smoothing keyed by spatial region (grid cell)
        # so that multiple people don't share identity history.
        self._histories: dict[tuple, deque] = {}

        # FPS tracking
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps = 0.0

        # Recent faces buffer (for the /faces REST endpoint)
        self._recent_faces: deque[DetectedFace] = deque(maxlen=50)
        self._lock = threading.Lock()

    def _get_face_key(self, x1: int, y1: int, x2: int, y2: int) -> tuple:
        """Map a bounding box to a grid cell for per-face history tracking."""
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # Quantise to 80-pixel grid cells so nearby detections share history
        return (cx // 80, cy // 80)

    # ── Public API ────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> ProcessedFrame:
        """Run the full pipeline on *frame* and return annotated result."""
        yolo = get_yolo()
        arcface = get_arcface()

        frame = cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))
        frame = enhance_frame(frame)

        detected_faces: List[DetectedFace] = []

        results = yolo(
            frame,
            conf=settings.YOLO_CONF_THRESHOLD,
            imgsz=settings.IMGSZ,
            device=_DEVICE,
            verbose=False,
        )

        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                if float(boxes.conf[i]) < settings.BOX_CONF_THRESHOLD:
                    continue

                cls = int(boxes.cls[i])
                label = yolo.names.get(cls, "")
                if label.lower() != "face":
                    continue

                x1, y1, x2, y2 = map(int, boxes.xyxy[i])

                # Expand bounding box by margin
                h, w = frame.shape[:2]
                x1 = max(0, x1 - settings.FACE_MARGIN)
                y1 = max(0, y1 - settings.FACE_MARGIN)
                x2 = min(w, x2 + settings.FACE_MARGIN)
                y2 = min(h, y2 + settings.FACE_MARGIN)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Pre-process face crop
                face = enhance_face(face)
                face_resized = cv2.resize(face, (settings.FACE_SIZE, settings.FACE_SIZE))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                # Compute embedding
                embedding = arcface.get_feat(face_rgb).flatten()
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    continue
                embedding = embedding / norm

                # Recognise
                identity, score = recognize_face(embedding)

                # Temporal smoothing (per-face, keyed by spatial region)
                face_key = self._get_face_key(x1, y1, x2, y2)
                if face_key not in self._histories:
                    self._histories[face_key] = deque(maxlen=settings.HISTORY_LEN)
                self._histories[face_key].append(identity)
                identity = max(set(self._histories[face_key]),
                               key=self._histories[face_key].count)

                # Draw on frame
                color = (0, 255, 0) if identity != "Unauthorized" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{identity} ({score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                detected = DetectedFace(
                    name=identity,
                    confidence=round(score, 4),
                    bbox=(x1, y1, x2, y2),
                )
                detected_faces.append(detected)

        # Update recent faces
        with self._lock:
            self._recent_faces.extend(detected_faces)

        # FPS calculation
        self._frame_count += 1
        if self._frame_count >= 10:
            now = time.time()
            self._fps = self._frame_count / (now - self._fps_time)
            self._fps_time = now
            self._frame_count = 0

        # Draw FPS on frame
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
