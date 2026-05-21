"""
MongoDB connection for SafeVision.

Credentials are read from the ``MONGO_URI`` environment variable — **never**
hardcoded.  Locally this comes from ``.env``; in production it is injected
from Google Secret Manager via the container environment.
"""

import logging

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
