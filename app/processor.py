"""
Frame processing pipeline for SafeVision.

Simplified stateless architecture: a single process(frame) method that runs
YOLO and ArcFace synchronously on every frame, eliminating tracking drift
and thread contention.
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
from app.recognition import recognize_face
from app.alerts import send_unauthorized_alert

# Lock to prevent PyTorch and ONNXRuntime from colliding on the GPU
_GPU_LOCK = threading.Lock()

logger = logging.getLogger("safevision")

@dataclass
class DetectedFace:
    """Represents a single recognized face in the current frame."""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class AIResult:
    """Contains all AI outputs for a given frame."""
    faces: List[DetectedFace] = field(default_factory=list)
    fps: float = 0.0

ARC_FACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(img, landmarks):
    # Use OpenCV's built-in Similarity Transform estimator instead of scikit-image
    M, _ = cv2.estimateAffinePartial2D(landmarks, ARC_FACE_SRC, method=cv2.LMEDS)
    if M is None:
        raise ValueError("Failed to estimate affine transform")
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return aligned

class FrameProcessor:
    """
    Stateless Frame Processor.

    Executes YOLO and ArcFace on every frame. No tracking, no background
    queues. Guarantees 100% synchronization between bounding boxes and
    identities.
    """

    def __init__(self):
        # FPS tracking
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps = 0.0

        # Recent faces buffer (for the /faces REST endpoint)
        self._recent_faces: deque[DetectedFace] = deque(maxlen=50)
        import collections
        # Stateful track ID cache mapping track_id -> (identity, confidence)
        self._track_cache = collections.OrderedDict()
        self._alerted_tracks = collections.OrderedDict()
        self._cache_max_size = 200

    def _update_lru(self, cache, key, value):
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        if len(cache) > self._cache_max_size:
            cache.popitem(last=False)

    def clear_cache(self):
        """Force clear the tracker cache (useful when DB updates)."""
        self._track_cache.clear()
        self._alerted_tracks.clear()

    def detect(self, raw_frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, int, np.ndarray]]]:
        """
        Stage 1: Detect and track faces using YOLO + ByteTrack.
        Returns:
            (raw_frame, boxes_with_ids)
        """
        yolo = get_yolo()
        # Apply full-frame low-light enhancement if enabled before running detection
        detect_frame = enhance_frame(raw_frame)
        results = yolo.track(
            detect_frame,
            persist=True,
            conf=settings.YOLO_CONF_THRESHOLD,
            iou=0.45,
            imgsz=settings.IMGSZ,
            device=_DEVICE,
            verbose=False,
        )
        
        h, w = raw_frame.shape[:2]
        boxes = []
        for r in results:
            if r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(r.boxes)
                
            for i in range(len(r.boxes)):
                if float(r.boxes.conf[i]) < settings.BOX_CONF_THRESHOLD:
                    continue
                cls = int(r.boxes.cls[i])
                if yolo.names.get(cls, "").lower() != "face":
                    continue
                x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
                x1 = max(0, x1 - settings.FACE_MARGIN)
                y1 = max(0, y1 - settings.FACE_MARGIN)
                x2 = min(w, x2 + settings.FACE_MARGIN)
                y2 = min(h, y2 + settings.FACE_MARGIN)
                kpts = None
                if hasattr(r, 'keypoints') and r.keypoints is not None:
                    # Keypoints might be empty if the model doesn't support them
                    if len(r.keypoints.xy) > i and r.keypoints.xy[i].shape[0] == 5:
                        kpts = r.keypoints.xy[i].cpu().numpy()
                boxes.append((x1, y1, x2, y2, track_ids[i], kpts))
                
        return raw_frame, boxes

    def recognize_faces(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int, int, np.ndarray]]) -> AIResult:
        """
        Stage 2: Recognize faces, skipping ArcFace for cached track IDs.
        Returns:
            AIResult
        """
        detected_faces: List[DetectedFace] = []
        valid_boxes = []
        valid_faces_rgb = []
        valid_track_ids = []

        for (x1, y1, x2, y2, track_id, kpts) in boxes:
            bbox = (x1, y1, x2, y2)
            
            # Check Cache First
            if track_id is not None and track_id in self._track_cache:
                identity, score = self._track_cache[track_id]
                # If we confidently know who this is, skip ArcFace!
                if identity not in ("Unauthorized", "Unknown", "Too Blurry"):
                    self._update_lru(self._track_cache, track_id, (identity, score))
                    detected_faces.append(DetectedFace(
                        name=identity,
                        confidence=score,
                        bbox=bbox,
                    ))
                    continue

            # Need to run ArcFace
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            if kpts is not None:
                try:
                    # Align face mathematically using 5 keypoints
                    face_112 = align_face(frame, kpts)
                except Exception as e:
                    logger.warning(f"Face alignment failed: {e}")
                    face_112 = cv2.resize(face_crop, (settings.FACE_SIZE, settings.FACE_SIZE))
            else:
                # Fallback to naive resize if YOLO didn't output keypoints
                face_112 = cv2.resize(face_crop, (settings.FACE_SIZE, settings.FACE_SIZE))

            blur_score = cv2.Laplacian(face_112, cv2.CV_64F).var()
            
            if blur_score < settings.BLUR_THRESHOLD:
                detected_faces.append(DetectedFace(
                    name="Too Blurry",
                    confidence=0.0,
                    bbox=bbox,
                ))
            else:
                face_enhanced = enhance_face(face_112)
                face_rgb = cv2.cvtColor(face_enhanced, cv2.COLOR_BGR2RGB)
                valid_faces_rgb.append(face_rgb)
                valid_boxes.append(bbox)
                valid_track_ids.append(track_id)

        if valid_faces_rgb:
            arcface = get_arcface()
            # Batch inference on all valid faces in a single ONNX call
            with _GPU_LOCK:
                embeddings = arcface.get_feats(valid_faces_rgb)

            for bbox, embedding, track_id in zip(valid_boxes, embeddings, valid_track_ids):
                identity, score = recognize_face(embedding)
                
                if identity == "Unauthorized":
                    # Only alert once per track ID to prevent disk exhaustion
                    if track_id is None or track_id not in self._alerted_tracks:
                        if track_id is not None:
                            self._update_lru(self._alerted_tracks, track_id, True)
                        bx1, by1, bx2, by2 = bbox
                        send_unauthorized_alert(score, bbox, frame[by1:by2, bx1:bx2].copy())
                elif identity not in ("Unknown", "Too Blurry") and track_id is not None:
                    # Successfully recognized an authorized person! Cache it.
                    self._update_lru(self._track_cache, track_id, (identity, round(score, 4)))

                detected_faces.append(DetectedFace(
                    name=identity,
                    confidence=round(score, 4),
                    bbox=bbox,
                ))

        # Update recent faces buffer
        self._recent_faces.extend(detected_faces)

        # FPS calculation
        self._frame_count += 1
        if self._frame_count >= 10:
            now = time.time()
            self._fps = self._frame_count / (now - self._fps_time)
            self._fps_time = now
            self._frame_count = 0

        return AIResult(faces=detected_faces, fps=round(self._fps, 2))

    @property
    def fps(self) -> float:
        return self._fps

    def get_recent_faces(self, limit: int = 10) -> List[dict]:
        items = list(self._recent_faces)
        # Return only the most recent `limit` entries
        return [
            {
                "name": f.name,
                "confidence": f.confidence,
                "bbox": f.bbox,
            }
            for f in items[-limit:]
        ]
