"""
Push notification alerts via Firebase Cloud Messaging (FCM).

Sends a real-time push notification to the mobile app when an unauthorized
face is detected by the SafeVision pipeline.
"""
import logging
import firebase_admin
from firebase_admin import credentials, messaging

from app.config import settings

logger = logging.getLogger("safevision")

_initialized = False

def init_firebase():
    """Initialize the Firebase Admin SDK if credentials are provided."""
    global _initialized
    if not _initialized and settings.FIREBASE_CREDENTIALS_PATH:
        try:
            cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
            firebase_admin.initialize_app(cred)
            _initialized = True
            logger.info("Firebase Cloud Messaging initialized.")
        except Exception as exc:
            logger.error("Failed to initialize Firebase: %s", exc)

import cv2
import uuid
import os
from datetime import datetime
import numpy as np

from app.database import async_alerts_collection

async def send_unauthorized_alert(confidence: float, bbox: tuple, face_img: np.ndarray) -> None:
    """
    Send an FCM notification when an unauthorized face is detected and save the photo to the VM.
    
    Parameters
    ----------
    confidence : float
        Recognition score (how confident we are it's unauthorized, or top score).
    bbox : tuple
        (x1, y1, x2, y2) of the detected face.
    face_img : np.ndarray
        The BGR image crop of the face.
    """
    # ── Save image to VM storage ───────────────────────────────────────
    filename = ""
    try:
        # Create directory just in case it doesn't exist yet inside container
        os.makedirs("/opt/safevision/unauthorized_faces", exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unauthorized_{timestamp_str}_{uuid.uuid4().hex[:8]}.jpg"
        image_path = os.path.join("/opt/safevision/unauthorized_faces", filename)
        
        cv2.imwrite(image_path, face_img)
        logger.info(f"Saved unauthorized face photo to {image_path}")

        # ── Save to Alerts Database ────────────────────────────────────────
        alert_doc = {
            "timestamp": datetime.now(),
            "confidence": confidence,
            "image_filename": filename,
            "type": "unauthorized"
        }
        await async_alerts_collection.insert_one(alert_doc)

    except Exception as exc:
        logger.error(f"Failed to save unauthorized alert: {exc}")

    if not _initialized or not settings.FCM_TOPIC:
        return
        
    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title="⚠️ Unauthorized Person Detected",
                body=f"Confidence: {confidence:.0%} — check the live stream.",
            ),
            data={
                "bbox": str(bbox),
                "confidence": str(confidence),
                "type": "unauthorized_access",
                "image_filename": filename
            },
            topic=settings.FCM_TOPIC,
        )
        messaging.send(message)
        logger.info("FCM alert sent for unauthorized detection.")
    except Exception as exc:
        logger.warning("FCM alert failed: %s", exc)
