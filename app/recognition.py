"""
Face recognition against MongoDB Atlas vector search.

Phase 4 update: Two-tier recognition with local FAISS cache.

Recognition now uses a two-tier approach:
  1. **FAISS local index** — instant (<0.1 ms) lookup against cached embeddings
  2. **MongoDB Atlas $vectorSearch** — fallback when FAISS is not loaded

Adaptive per-identity thresholds are preserved: at startup,
``_IDENTITY_THRESHOLDS`` is populated by computing intra-class pairwise
cosine similarities for every enrolled person.

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


def get_all_thresholds() -> Dict[str, float]:
    """Return a copy of all per-identity thresholds (for FAISS search)."""
    with _threshold_lock:
        return dict(_IDENTITY_THRESHOLDS)


# Load thresholds once at import time (non-blocking — fails gracefully)
try:
    reload_thresholds()
except Exception as exc:
    logger.warning("Could not load adaptive thresholds at startup: %s", exc)


# ---------------------------------------------------------------------------
# Local FAISS index (loaded once, rebuilt on enrollment changes)
# ---------------------------------------------------------------------------
from app.faiss_index import LocalFaceIndex

_faiss_index = LocalFaceIndex()

def get_faiss_index() -> LocalFaceIndex:
    """Return the singleton FAISS index instance."""
    return _faiss_index


def load_faiss_index() -> int:
    """Load/rebuild the FAISS index from MongoDB. Returns count of embeddings loaded."""
    count = _faiss_index.load_from_mongodb(faces_collection)
    logger.info("FAISS index ready — %d embeddings cached locally", count)
    return count


# ---------------------------------------------------------------------------
# Recognition — two-tier: FAISS local → MongoDB fallback
# ---------------------------------------------------------------------------
def recognize_face(
    embedding: np.ndarray,
    threshold: float | None = None,
) -> Tuple[str, float]:
    """
    Search for the closest face embedding using two-tier lookup.

    Tier 1: Local FAISS index (sub-millisecond).
    Tier 2: MongoDB Atlas $vectorSearch (network round-trip fallback).

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
    # ── Tier 1: FAISS local index (instant) ──────────────────────────────
    if _faiss_index.is_loaded:
        name, score = _faiss_index.search(
            embedding,
            threshold=threshold if threshold is not None else settings.RECOG_THRESHOLD,
            per_identity_thresholds=get_all_thresholds() if threshold is None else None,
            top_k=settings.DB_TOP_K,
        )
        if name is not None:
            return name, score
        else:
            return "Unauthorized", score

    # ── Tier 2: MongoDB Atlas $vectorSearch (network fallback) ───────────
    try:
        cursor = faces_collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": embedding.flatten().tolist(),
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
            raw_score = match.get("score", 0.0)
            # MongoDB Atlas $vectorSearch 'cosine' metric returns (1 + cosine) / 2
            # We must convert it back to raw cosine similarity to match FAISS and our thresholds
            score = (2.0 * raw_score) - 1.0
            
            if score > scores_by_name.get(name, -1.0):
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
        return "Unauthorized", 0.0

from app.database import async_faces_collection

async def async_recognize_face(
    embedding: np.ndarray,
    threshold: float | None = None,
) -> Tuple[str, float]:
    """
    Asynchronous version of recognize_face using two-tier lookup.

    Tier 1: Local FAISS index (instant, no await needed).
    Tier 2: MongoDB Atlas $vectorSearch via Motor (async network fallback).
    """
    # ── Tier 1: FAISS local index (instant — no I/O) ─────────────────────
    if _faiss_index.is_loaded:
        name, score = _faiss_index.search(
            embedding,
            threshold=threshold if threshold is not None else settings.RECOG_THRESHOLD,
            per_identity_thresholds=get_all_thresholds() if threshold is None else None,
            top_k=settings.DB_TOP_K,
        )
        if name is not None:
            return name, score
        else:
            return "Unauthorized", score

    # ── Tier 2: MongoDB Atlas $vectorSearch (async network fallback) ─────
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": embedding.flatten().tolist(),
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
        ]
        
        cursor = async_faces_collection.aggregate(pipeline)
        results = await cursor.to_list(length=settings.DB_TOP_K)

        if not results:
            return "Unauthorized", 0.0

        scores_by_name: Dict[str, float] = {}
        for match in results:
            name = match.get("name", "Unknown")
            raw_score = match.get("score", 0.0)
            score = (2.0 * raw_score) - 1.0
            if score > scores_by_name.get(name, -1.0):
                scores_by_name[name] = score

        best_name = max(scores_by_name, key=scores_by_name.__getitem__)
        best_score = scores_by_name[best_name]

        effective_threshold = threshold if threshold is not None else get_threshold(best_name)

        if best_score < effective_threshold:
            logger.debug(
                "[Async] Best match '%s' (score=%.3f) below threshold %.3f — Unauthorized",
                best_name, best_score, effective_threshold,
            )
            return "Unauthorized", best_score

        return best_name, best_score

    except Exception as exc:
        logger.error("[Async] MongoDB vector search error: %s", exc)
        return "Unauthorized", 0.0
