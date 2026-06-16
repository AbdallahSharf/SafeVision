"""
Local FAISS index for instant face matching.

Maintains an in-memory FAISS IndexFlatIP (inner product on L2-normalised
vectors = cosine similarity) synchronised with the MongoDB enrollment
collection.  This eliminates the 20–100 ms network round-trip to MongoDB
Atlas for every recognition query.

Usage:
    index = LocalFaceIndex()
    index.load_from_mongodb(faces_collection)
    name, score = index.search(embedding, threshold=0.6)
"""

import logging
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("safevision")

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss-cpu not installed — LocalFaceIndex will fall back to numpy. "
        "Install with: pip install faiss-cpu"
    )


class LocalFaceIndex:
    """
    Local FAISS index for sub-millisecond face matching.

    Thread-safe.  All mutations are protected by a lock, and reads use
    a snapshot pattern to avoid holding the lock during search.
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self._lock = threading.Lock()
        self._names: list[str] = []
        self._embeddings: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._loaded = False
        self._last_sync: float = 0.0

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dimension)
        else:
            self._index = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def total_embeddings(self) -> int:
        with self._lock:
            return len(self._names)

    def load_from_mongodb(self, faces_collection) -> int:
        """
        Pull all enrolled embeddings from MongoDB into the local index.

        Parameters
        ----------
        faces_collection : pymongo.collection.Collection
            The sync MongoDB collection containing enrolled faces.

        Returns
        -------
        int
            Number of embeddings loaded.
        """
        try:
            docs = list(faces_collection.find({}, {"name": 1, "embedding": 1}))
        except Exception as exc:
            logger.error("Failed to load embeddings from MongoDB: %s", exc)
            return 0

        if not docs:
            logger.warning("No enrolled faces found in MongoDB — FAISS index is empty")
            self._loaded = True
            self._last_sync = time.time()
            return 0

        names: list[str] = []
        embeddings: list[np.ndarray] = []

        for doc in docs:
            emb = np.array(doc["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm  # L2 normalise for cosine similarity via IP
            names.append(doc["name"])
            embeddings.append(emb)

        matrix = np.stack(embeddings).astype(np.float32)

        with self._lock:
            self._names = names
            self._embeddings = matrix

            if _FAISS_AVAILABLE:
                self._index = faiss.IndexFlatIP(self.dimension)
                self._index.add(matrix)

            self._loaded = True
            self._last_sync = time.time()

        logger.info(
            "FAISS index loaded: %d embeddings for %d identities",
            len(names), len(set(names)),
        )
        return len(names)

    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6,
        per_identity_thresholds: Optional[Dict[str, float]] = None,
        top_k: int = 5,
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching identity for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            L2-normalised 512-d face embedding.
        threshold : float
            Global recognition threshold (fallback).
        per_identity_thresholds : dict, optional
            Per-identity adaptive thresholds.
        top_k : int
            Number of nearest neighbours to retrieve.

        Returns
        -------
        (name, score) or (None, 0.0) if index is not loaded.
            name is "Unauthorized" if no match exceeds threshold.
        """
        if not self._loaded:
            return None, 0.0

        with self._lock:
            n_total = len(self._names)
            if n_total == 0:
                return "Unauthorized", 0.0
            # Snapshot for thread safety
            names_snapshot = list(self._names)

        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, n_total)

        if _FAISS_AVAILABLE and self._index is not None:
            scores, indices = self._index.search(query, k)
            results = [
                (names_snapshot[idx], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx != -1
            ]
        else:
            # Numpy fallback: dot product (= cosine similarity on L2-normed vectors)
            with self._lock:
                sims = (self._embeddings @ query.T).flatten()
            top_indices = np.argsort(-sims)[:k]
            results = [
                (names_snapshot[idx], float(sims[idx]))
                for idx in top_indices
            ]

        if not results:
            return "Unauthorized", 0.0

        # Group by name, keep highest score per person
        scores_by_name: Dict[str, float] = {}
        for name, score in results:
            if score > scores_by_name.get(name, 0.0):
                scores_by_name[name] = score

        best_name = max(scores_by_name, key=scores_by_name.get)
        best_score = scores_by_name[best_name]

        # Apply per-identity adaptive threshold if available
        effective_threshold = threshold
        if per_identity_thresholds and best_name in per_identity_thresholds:
            effective_threshold = per_identity_thresholds[best_name]

        if best_score >= effective_threshold:
            return best_name, best_score

        return "Unauthorized", best_score

    def rebuild(self, faces_collection) -> int:
        """
        Rebuild the index from MongoDB.  Call after enrolling or deleting faces.

        This is an alias for load_from_mongodb() with clearer intent.
        """
        return self.load_from_mongodb(faces_collection)
