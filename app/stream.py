"""
Threaded RTSP video stream reader.

Runs a background thread that continuously reads frames from the RTSP
camera and keeps a bounded queue of the latest frames.  Consumers call
``read()`` to get the most recent frame.
"""

from app.config import settings
import logging
import queue
import threading
import time
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
import cv2

logger = logging.getLogger("safevision")


class VideoStream:
    """Non-blocking RTSP reader with automatic reconnection."""

    def __init__(
        self,
        rtsp_url: str | None = None,
        queue_size: int | None = None,
    ):
        self.rtsp_url = rtsp_url or settings.RTSP_URL
        self._queue_size = queue_size or settings.QUEUE_SIZE

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            logger.warning("Cannot open RTSP stream at startup: %s. Will keep retrying.", self.rtsp_url)

        self.q: queue.Queue = queue.Queue(maxsize=self._queue_size)
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        logger.info("VideoStream started — %s", self.rtsp_url)

    # ── Background reader ─────────────────────────────────────────────────
    def _update(self) -> None:
        while self.running:
            try:
                if not self.cap.isOpened():
                    logger.warning("Stream lost — reconnecting …")
                    self.cap.open(self.rtsp_url)
                    time.sleep(1)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                # Drop oldest frame if the consumer is too slow
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass

                self.q.put(frame)

            except Exception as exc:
                logger.error("Stream thread error: %s", exc)
                time.sleep(1)

    # ── Public API ────────────────────────────────────────────────────────
    def read(self, timeout: float | None = None):
        """Return the latest frame, or ``None`` on timeout."""
        if timeout is None:
            timeout = settings.QUEUE_TIMEOUT
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            logger.warning("Frame queue timed out — stream may be stalled.")
            return None

    def is_alive(self) -> bool:
        """Return ``True`` if the background thread is still running."""
        return self._thread.is_alive() and self.running

    def stop(self) -> None:
        """Stop reading and release the capture device."""
        self.running = False
        self._thread.join(timeout=3)
        self.cap.release()
        logger.info("VideoStream stopped.")
