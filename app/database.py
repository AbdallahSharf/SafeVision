"""
MongoDB connection for SafeVision.

Credentials are read from the ``MONGO_URI`` environment variable — **never**
hardcoded.  Locally this comes from ``.env``; in production it is injected
from Google Secret Manager via the container environment.
"""

import logging
from typing import Dict

import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from app.config import settings

logger = logging.getLogger("safevision")

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
try:
    client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
    # Force a connection test so we fail fast on bad credentials / network
    client.admin.command("ping")
    logger.info("Connected to MongoDB Atlas successfully.")
except ConnectionFailure as exc:
    logger.critical("Cannot connect to MongoDB: %s", exc)
    raise SystemExit(1) from exc

# ---------------------------------------------------------------------------
# Database & collection
# ---------------------------------------------------------------------------
db = client["unidbb"]
faces_collection = db["faces"]


# ---------------------------------------------------------------------------
# Adaptive threshold calibration
# ---------------------------------------------------------------------------
def compute_identity_thresholds() -> Dict[str, float]:
    """
    Compute a per-identity recognition threshold calibrated from enrollment data.

    For each enrolled person, we compute the pairwise cosine similarities
    between all their stored embeddings.  The threshold is set to:

        mean(intra-class similarities) - 1.5 * std(intra-class similarities)

    This means:
    - People with very consistent embeddings (good lighting, frontal) get a
      TIGHTER threshold → harder to spoof with a lookalike.
    - People with variable embeddings (glasses, angles) get a LOOSER threshold
      → they are still recognised despite variation.

    Falls back to ``settings.RECOG_THRESHOLD`` for any person with fewer than
    3 enrolled embeddings (not enough data to calibrate).

    Returns
    -------
    dict mapping name → calibrated threshold (float in [0, 1])
    """
    pipeline = [
        {"$group": {"_id": "$name", "embeddings": {"$push": "$embedding"}}},
    ]

    thresholds: Dict[str, float] = {}
    try:
        for doc in faces_collection.aggregate(pipeline):
            name = doc["_id"]
            raw_embeddings = doc["embeddings"]

            if len(raw_embeddings) < 3:
                # Not enough samples — fall back to global default
                logger.debug(
                    "Identity '%s' has only %d embedding(s) — using global threshold.",
                    name, len(raw_embeddings),
                )
                continue

            embs = np.array(raw_embeddings, dtype=np.float32)

            # L2-normalise each row (in case any were stored un-normalised)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embs = embs / norms

            # Pairwise cosine similarities (all upper-triangle pairs)
            sim_matrix = embs @ embs.T                          # (N, N)
            n = len(embs)
            upper_idx = np.triu_indices(n, k=1)                 # exclude diagonal
            similarities = sim_matrix[upper_idx]

            if len(similarities) == 0:
                continue

            mean_sim = float(similarities.mean())
            std_sim  = float(similarities.std())

            # threshold = mean - 1.5σ, clamped between 0.3 and 0.95
            calibrated = float(np.clip(mean_sim - 1.5 * std_sim, 0.30, 0.95))
            thresholds[name] = calibrated

            logger.info(
                "Calibrated threshold for '%s': %.3f "
                "(mean_sim=%.3f, std=%.3f, n=%d embeddings)",
                name, calibrated, mean_sim, std_sim, n,
            )

    except Exception as exc:
        logger.warning("Could not compute adaptive thresholds: %s — using global default.", exc)

    return thresholds
