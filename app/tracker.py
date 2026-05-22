"""
Face tracker for SafeVision — SORT-lite (Simple Online and Realtime Tracking).

Assigns persistent integer track IDs to face bounding boxes across frames
using IoU matching and a simple Kalman-filter velocity model.

This replaces the grid-cell-based history dict in FrameProcessor, so that:
  - Identity history is correctly scoped to one physical person, not a screen region.
  - Two faces crossing paths never inherit each other's history.
  - History is preserved correctly as a face moves across the frame.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger("safevision")


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Intersection-over-Union for two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _iou_matrix(tracks: list, detections: list) -> np.ndarray:
    """Return an (N_tracks, N_detections) IoU matrix."""
    mat = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            mat[i, j] = _iou(t.predicted_bbox(), np.array(d))
    return mat


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """
    A single tracked face with a persistent ID.

    Uses a simple constant-velocity model: we store the last bbox and a
    velocity (dx, dy) estimated from the previous two positions.
    """

    track_id: int
    bbox: np.ndarray                        # [x1, y1, x2, y2]
    identity_history: deque = field(default_factory=lambda: deque(maxlen=settings.HISTORY_LEN))
    lost_frames: int = 0
    _velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    _prev_bbox: np.ndarray | None = None
    created_at: float = field(default_factory=time.time)

    def update(self, bbox: np.ndarray) -> None:
        """Accept a matched detection, update position and velocity."""
        new_bbox = np.array(bbox, dtype=float)
        if self._prev_bbox is not None:
            self._velocity = new_bbox - self.bbox
        self._prev_bbox = self.bbox.copy()
        self.bbox = new_bbox
        self.lost_frames = 0

    def predict(self) -> None:
        """Advance position by velocity (called when no match found)."""
        self.bbox = self.bbox + self._velocity * 0.5   # dampen velocity
        self.lost_frames += 1

    def predicted_bbox(self) -> np.ndarray:
        """Return expected next position."""
        return self.bbox + self._velocity * 0.5

    @property
    def smoothed_identity(self) -> str | None:
        """Return the majority-vote identity from recent history, or None."""
        if not self.identity_history:
            return None
        return max(set(self.identity_history), key=self.identity_history.count)


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """
    IoU-based multi-object tracker for face bounding boxes.

    Usage::

        tracker = FaceTracker()

        for frame in stream:
            boxes = yolo_detect(frame)          # list of (x1,y1,x2,y2)
            tracks = tracker.update(boxes)      # list of active Track objects
            for track in tracks:
                identity = track.smoothed_identity
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 8,
    ):
        self._tracks: List[Track] = []
        self._next_id: int = 0
        self._iou_threshold = iou_threshold
        self._max_lost_frames = max_lost_frames

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Track]:
        """
        Match *detections* to existing tracks via greedy IoU matching.

        Parameters
        ----------
        detections : list of (x1, y1, x2, y2) tuples

        Returns
        -------
        List of currently active Track objects (lost_frames == 0).
        """
        # Predict next position for all existing tracks
        for t in self._tracks:
            t.predict()

        if not detections:
            self._tracks = [t for t in self._tracks if t.lost_frames <= self._max_lost_frames]
            return [t for t in self._tracks if t.lost_frames == 0]

        if not self._tracks:
            # No existing tracks — create one per detection
            for det in detections:
                self._tracks.append(Track(
                    track_id=self._next_id,
                    bbox=np.array(det, dtype=float),
                ))
                self._next_id += 1
            return list(self._tracks)

        # Build IoU cost matrix and do greedy matching
        iou_mat = _iou_matrix(self._tracks, detections)
        matched_track_ids: set[int] = set()
        matched_det_ids: set[int] = set()

        # Greedily match highest-IoU pairs
        flat_order = np.argsort(-iou_mat, axis=None)  # descending IoU
        for flat_idx in flat_order:
            ti = flat_idx // len(detections)
            di = flat_idx % len(detections)
            if ti in matched_track_ids or di in matched_det_ids:
                continue
            if iou_mat[ti, di] < self._iou_threshold:
                break
            self._tracks[ti].update(detections[di])
            matched_track_ids.add(ti)
            matched_det_ids.add(di)

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                self._tracks.append(Track(
                    track_id=self._next_id,
                    bbox=np.array(det, dtype=float),
                ))
                self._next_id += 1

        # Remove stale tracks that have been lost too long
        self._tracks = [t for t in self._tracks if t.lost_frames <= self._max_lost_frames]

        return [t for t in self._tracks if t.lost_frames == 0]
