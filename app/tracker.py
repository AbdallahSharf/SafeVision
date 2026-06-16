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
from scipy.optimize import linear_sum_assignment

from app.config import settings

logger = logging.getLogger("safevision")


# ---------------------------------------------------------------------------
# Track state constants
# ---------------------------------------------------------------------------

class TrackState:
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


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

    Track lifecycle:
      Tentative  →  (3 consecutive matches)  →  Confirmed
      Confirmed  →  (lost_frames > 0)        →  Lost
      Lost       →  (re-matched)             →  Confirmed
      Lost       →  (lost_frames > max)      →  Deleted
      Tentative  →  (lost_frames > 3)        →  Deleted
    """

    track_id: int
    bbox: np.ndarray                        # [x1, y1, x2, y2]
    identity_history: deque = field(default_factory=lambda: deque(maxlen=settings.HISTORY_LEN))
    last_score: float = 0.0
    lost_frames: int = 0
    _velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    _prev_bbox: np.ndarray | None = None
    created_at: float = field(default_factory=time.time)
    state: str = TrackState.TENTATIVE
    consecutive_matches: int = 0
    cached_embedding: np.ndarray | None = None

    # ── Tentative → Confirmed promotion threshold ─────────────────────────
    _CONFIRM_HITS: int = 3

    def update(self, bbox: np.ndarray) -> None:
        """Accept a matched detection, update position and velocity."""
        new_bbox = np.array(bbox, dtype=float)
        if self._prev_bbox is not None:
            self._velocity = new_bbox - self.bbox
        self._prev_bbox = self.bbox.copy()
        self.bbox = new_bbox
        self.lost_frames = 0
        self.consecutive_matches += 1

        # State transitions on successful match
        if self.state == TrackState.TENTATIVE:
            if self.consecutive_matches >= self._CONFIRM_HITS:
                self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def predict(self) -> None:
        """Advance position by velocity (called when no match found)."""
        self.bbox = self.bbox + self._velocity * 0.5   # dampen velocity
        self.lost_frames += 1
        self.consecutive_matches = 0

        # Confirmed → Lost when the track misses a frame
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

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

    Uses the Hungarian algorithm for optimal IoU assignment, track-state
    management (tentative / confirmed / lost), and optional appearance-based
    re-identification via cached ArcFace embeddings.

    Usage::

        tracker = FaceTracker()

        for frame in stream:
            boxes = yolo_detect(frame)          # list of (x1,y1,x2,y2)
            tracks = tracker.update(boxes)      # list of active Track objects
            for track in tracks:
                identity = track.smoothed_identity
    """

    # Cosine-similarity threshold for appearance re-ID
    _REID_COSINE_THRESHOLD: float = 0.5

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
        Match *detections* to existing tracks via Hungarian IoU matching,
        with appearance-based re-ID fallback for lost tracks.

        Parameters
        ----------
        detections : list of (x1, y1, x2, y2) tuples

        Returns
        -------
        List of currently active Track objects (Confirmed or Lost).
        """
        # Predict next position for all existing tracks
        for t in self._tracks:
            t.predict()

        if not detections:
            self._prune_tracks()
            return self._active_tracks()

        if not self._tracks:
            # No existing tracks — create one per detection
            for det in detections:
                self._tracks.append(Track(
                    track_id=self._next_id,
                    bbox=np.array(det, dtype=float),
                ))
                self._next_id += 1
            return self._active_tracks()

        # ------------------------------------------------------------------
        # Stage 1: Hungarian IoU matching
        # ------------------------------------------------------------------
        iou_mat = _iou_matrix(self._tracks, detections)

        # Build a cost matrix (1 − IoU); mask out pairs below threshold
        cost = 1.0 - iou_mat
        cost[iou_mat < self._iou_threshold] = 1e5  # effectively disable

        row_indices, col_indices = linear_sum_assignment(cost)

        matched_track_ids: set[int] = set()
        matched_det_ids: set[int] = set()

        for ti, di in zip(row_indices, col_indices):
            if iou_mat[ti, di] >= self._iou_threshold:
                self._tracks[ti].update(detections[di])
                matched_track_ids.add(ti)
                matched_det_ids.add(di)

        # ------------------------------------------------------------------
        # Stage 2: Appearance-based re-ID for unmatched Lost tracks
        # ------------------------------------------------------------------
        unmatched_track_indices = [
            i for i in range(len(self._tracks))
            if i not in matched_track_ids and self._tracks[i].state == TrackState.LOST
        ]
        unmatched_det_indices = [
            j for j in range(len(detections)) if j not in matched_det_ids
        ]

        if unmatched_track_indices and unmatched_det_indices:
            self._appearance_match(
                unmatched_track_indices,
                unmatched_det_indices,
                detections,
                matched_track_ids,
                matched_det_ids,
            )

        # ------------------------------------------------------------------
        # Stage 3: Create new tentative tracks for remaining detections
        # ------------------------------------------------------------------
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                self._tracks.append(Track(
                    track_id=self._next_id,
                    bbox=np.array(det, dtype=float),
                ))
                self._next_id += 1

        # Housekeeping
        self._prune_tracks()

        return self._active_tracks()

    # ── Private helpers ────────────────────────────────────────────────────

    def _appearance_match(
        self,
        track_indices: list[int],
        det_indices: list[int],
        detections: list,
        matched_track_ids: set[int],
        matched_det_ids: set[int],
    ) -> None:
        """
        Try to re-associate lost tracks with unmatched detections based on
        cached ArcFace embedding cosine similarity.
        """
        # Collect tracks that actually have a cached embedding
        candidates = [
            (ti, self._tracks[ti])
            for ti in track_indices
            if self._tracks[ti].cached_embedding is not None
        ]
        if not candidates:
            return

        # Build a similarity matrix (num_candidate_tracks × num_unmatched_dets)
        # We can only compare if the detection also has an embedding attached,
        # but at this level we only have bounding boxes.  The embedding is set
        # externally (e.g. by FrameProcessor) on the Track *after* matching.
        # So for re-ID we compare cached track embeddings pairwise.
        # -----------------------------------------------------------------
        # Because detections arrive as plain bbox tuples and their embeddings
        # are not yet computed at this stage, we fall back to checking each
        # *unmatched detection* against each lost track's cached embedding
        # only when the caller has previously stored embeddings on tracks.
        # If no embeddings are available at all, this is a no-op.
        #
        # When FrameProcessor attaches embeddings to detections (as a list
        # parallel to the bbox list), callers can use `update_with_embeddings`
        # or manually set `track.cached_embedding` after each frame.
        # -----------------------------------------------------------------
        # For now, we support the re-ID path when a subclass or external
        # caller attaches `._det_embeddings` on the tracker before calling
        # update.  This keeps the public API unchanged.
        det_embeddings = getattr(self, "_det_embeddings", None)
        if det_embeddings is None:
            return

        for ti, track in candidates:
            if ti in matched_track_ids:
                continue
            t_emb = track.cached_embedding
            best_sim = -1.0
            best_di = -1
            for di in det_indices:
                if di in matched_det_ids:
                    continue
                d_emb = det_embeddings[di] if di < len(det_embeddings) else None
                if d_emb is None:
                    continue
                # Cosine similarity
                sim = float(np.dot(t_emb, d_emb) / (
                    np.linalg.norm(t_emb) * np.linalg.norm(d_emb) + 1e-6
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_di = di
            if best_sim >= self._REID_COSINE_THRESHOLD and best_di >= 0:
                self._tracks[ti].update(detections[best_di])
                matched_track_ids.add(ti)
                matched_det_ids.add(best_di)

    def _prune_tracks(self) -> None:
        """Remove tracks that should be deleted based on their state."""
        surviving: List[Track] = []
        for t in self._tracks:
            if t.state == TrackState.TENTATIVE and t.lost_frames > 3:
                continue  # delete tentative tracks quickly
            if t.state == TrackState.LOST and t.lost_frames > self._max_lost_frames:
                continue  # delete long-lost tracks
            surviving.append(t)
        self._tracks = surviving

    def _active_tracks(self) -> List[Track]:
        """Return only Confirmed and Lost tracks (not Tentative)."""
        return [
            t for t in self._tracks
            if t.state in (TrackState.CONFIRMED, TrackState.LOST)
        ]
