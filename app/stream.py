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
        # ── Critical: kill OpenCV's internal frame buffer ──────────────────
        # By default OpenCV buffers 4–10 decoded frames internally, creating
        # 130–330 ms of hidden latency even before our queue code runs.
        # Setting BUFFERSIZE=1 means only the single most-recent decoded frame
        # is kept; older frames are discarded immediately.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            logger.warning("Cannot open RTSP stream at startup: %s. Will keep retrying.", self.rtsp_url)

        self.q: queue.Queue = queue.Queue(maxsize=self._queue_size)
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        logger.info("VideoStream started — %s", self.rtsp_url)

    # ── Background reader ─────────────────────────────────────────────────
    def _update(self) -> None:
        fail_count = 0
        while self.running:
            try:
                if not self.cap.isOpened():
                    logger.warning("Stream lost — reconnecting …")
                    self.cap.open(self.rtsp_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # re-apply after reconnect
                    time.sleep(1)
                    fail_count = 0
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count > 20:
                        logger.warning("Stream read failed %d times. Forcing reconnection.", fail_count)
                        self.cap.release()
                        fail_count = 0
                    time.sleep(0.05)
                    continue

                fail_count = 0

                # Fast path: try to enqueue immediately (one lock acquisition)
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    # Queue full — drain oldest stale frame, then put the fresh one
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                    self.q.put_nowait(frame)

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

    def read_latest(self):
        """
        Return the single most-recent frame without blocking.

        Drains the entire queue and returns only the last item, discarding
        any intermediate frames that piled up.  Returns ``None`` if the
        queue is currently empty.
        """
        latest = None
        while True:
            try:
                latest = self.q.get_nowait()
            except queue.Empty:
                break
        return latest

    def is_alive(self) -> bool:
        """Return ``True`` if the background thread is still running."""
        return self._thread.is_alive() and self.running

    def stop(self) -> None:
        """Stop reading and release the capture device."""
        self.running = False
        self._thread.join(timeout=3)
        self.cap.release()
        logger.info("VideoStream stopped.")
