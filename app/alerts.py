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

def send_unauthorized_alert(confidence: float, bbox: tuple) -> None:
    """
    Send an FCM notification when an unauthorized face is detected.
    
    Parameters
    ----------
    confidence : float
        Recognition score (how confident we are it's unauthorized, or top score).
    bbox : tuple
        (x1, y1, x2, y2) of the detected face.
    """
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
                "type": "unauthorized_access"
            },
            topic=settings.FCM_TOPIC,
        )
        messaging.send(message)
        logger.info("FCM alert sent for unauthorized detection.")
    except Exception as exc:
        logger.warning("FCM alert failed: %s", exc)
