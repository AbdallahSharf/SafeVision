"""
Face recognition against MongoDB Atlas vector search.
"""

import logging
from typing import Tuple

import numpy as np

from app.config import settings
from app.database import faces_collection

logger = logging.getLogger("safevision")


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
        Minimum similarity score.  Defaults to ``settings.RECOG_THRESHOLD``.

    Returns
    -------
    (name, score)
        ``name`` is ``'Unauthorized'`` when the best score is below *threshold*.
    """
    if threshold is None:
        threshold = settings.RECOG_THRESHOLD

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
        scores_by_name: dict[str, float] = {}
        for match in results:
            name = match.get("name", "Unknown")
            score = match.get("score", 0.0)
            if score > scores_by_name.get(name, 0.0):
                scores_by_name[name] = score

        best_name = max(scores_by_name, key=scores_by_name.__getitem__)
        best_score = scores_by_name[best_name]

        if best_score < threshold:
            return "Unauthorized", best_score

        return best_name, best_score

    except Exception as exc:
        logger.error("MongoDB vector search error: %s", exc)
        return "Error", 0.0
