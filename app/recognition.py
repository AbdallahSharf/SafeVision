"""
Face recognition against MongoDB Atlas vector search.

Phase 3 update: Adaptive per-identity thresholds.

At startup, ``_IDENTITY_THRESHOLDS`` is populated by computing intra-class
pairwise cosine similarities for every enrolled person.  During live
recognition, the per-identity threshold is used instead of the global
``settings.RECOG_THRESHOLD``, giving tighter gates for consistent faces and
looser gates for faces enrolled under variable conditions.

Call ``reload_thresholds()`` to recompute after new faces are enrolled
without restarting the server.
"""

import logging
import threading
from typing import Dict, Tuple

import numpy as np

from app.config import settings
from app.database import faces_collection, compute_identity_thresholds

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Adaptive threshold cache
# ---------------------------------------------------------------------------
_IDENTITY_THRESHOLDS: Dict[str, float] = {}
_threshold_lock = threading.Lock()


def reload_thresholds() -> None:
    """
    Recompute per-identity thresholds from the current enrollment data.

    Thread-safe.  Call this after enrolling new faces so the live system
    picks up the updated calibration without a server restart.
    """
    global _IDENTITY_THRESHOLDS
    new_thresholds = compute_identity_thresholds()
    with _threshold_lock:
        _IDENTITY_THRESHOLDS = new_thresholds
    logger.info(
        "Adaptive thresholds reloaded — %d identities calibrated.",
        len(new_thresholds),
    )


def get_threshold(name: str) -> float:
    """
    Return the recognition threshold for *name*.

    Uses the calibrated per-identity threshold if available, otherwise
    falls back to the global ``settings.RECOG_THRESHOLD``.
    """
    with _threshold_lock:
        return _IDENTITY_THRESHOLDS.get(name, settings.RECOG_THRESHOLD)


# Load thresholds once at import time (non-blocking — fails gracefully)
try:
    reload_thresholds()
except Exception as exc:
    logger.warning("Could not load adaptive thresholds at startup: %s", exc)


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def recognize_face(
    embedding: np.ndarray,
    threshold: float | None = None,
) -> Tuple[str, float]:
    """
    Search MongoDB for the closest face embedding.

    Parameters
    ----------
    embedding : np.ndarray
        L2-normalised 512-d face embedding.
    threshold : float, optional
        Override threshold.  When *None* (the default), the per-identity
        adaptive threshold is used for the best match, falling back to
        ``settings.RECOG_THRESHOLD`` if no calibration data exists.

    Returns
    -------
    (name, score)
        ``name`` is ``'Unauthorized'`` when the best score is below
        the effective threshold.
    """
    try:
        cursor = faces_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": embedding.tolist(),
                    "path": "embedding",
                    "numCandidates": settings.DB_NUM_CANDIDATES,
                    "limit": settings.DB_TOP_K,
                }
            },
            {
                "$project": {
                    "name": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ])

        results = list(cursor)

        if not results:
            return "Unauthorized", 0.0

        # Group by name, keep highest score per person
        scores_by_name: Dict[str, float] = {}
        for match in results:
            name = match.get("name", "Unknown")
            score = match.get("score", 0.0)
            if score > scores_by_name.get(name, 0.0):
                scores_by_name[name] = score

        best_name = max(scores_by_name, key=scores_by_name.__getitem__)
        best_score = scores_by_name[best_name]

        # Use explicit override if provided, else per-identity adaptive threshold
        effective_threshold = threshold if threshold is not None else get_threshold(best_name)

        if best_score < effective_threshold:
            logger.debug(
                "Best match '%s' (score=%.3f) below threshold %.3f — Unauthorized",
                best_name, best_score, effective_threshold,
            )
            return "Unauthorized", best_score

        return best_name, best_score

    except Exception as exc:
        logger.error("MongoDB vector search error: %s", exc)
        return "Error", 0.0
