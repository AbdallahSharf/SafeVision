"""
Threaded RTSP video stream reader.

Runs a background thread that continuously reads frames from the RTSP
camera and exposes only the single latest frame via read_latest().

Design principles for zero-delay streaming:
  - One background thread owns the VideoCapture object exclusively.
  - Frames are pre-scaled to the target stream resolution here, once,
    so every downstream consumer receives a small frame — no repeated
    large-buffer copies downstream.
  - Reconnection is instant: the moment cap.read() fails, the capture
    is released and re-opened immediately.
  - FFmpeg options are passed via proper OpenCV CAP_PROP_* calls,
    not env-vars (which are unreliable across OpenCV versions).
  - No internal queue — _latest_frame is a single atomic reference.
    Python GIL makes one-reference assignment atomic, so no lock needed.
"""

from app.config import settings
import logging
import threading
import time
import os

# OpenCV environment variables are now set at the top of api.py BEFORE cv2 is imported

import cv2

logger = logging.getLogger("safevision")


class VideoStream:
    """Non-blocking RTSP reader with automatic reconnection and pre-scaling."""

    def __init__(
        self,
        rtsp_url: str | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
    ):
        self.rtsp_url = rtsp_url or settings.RTSP_URL
        # Pre-scale to stream output resolution
        self._target_w = target_width or settings.FRAME_WIDTH
        self._target_h = target_height or settings.FRAME_HEIGHT

        # Latest frame — written by reader thread, read by consumers (atomic via GIL)
        self._latest_frame = None
        self._last_decode_time = 0.0
        self.frame_ready_event = threading.Event()

        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True, name="StreamReader")
        self._thread.start()
        logger.info("VideoStream started — %s", self.rtsp_url)

    # ── Private: open capture ───────────────────────────────────────────────
    def _open_capture(self) -> cv2.VideoCapture:
        """Open the RTSP stream with TCP transport (set via env var above)."""
        # FFMPEG optimizations for ultra-low latency RTSP/HTTP
        # stimeout and timeout set to 5000000 (5 seconds in microseconds) prevent infinite hanging when Wi-Fi completely dies.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|recv_buffer_size;65536|stimeout;5000000|timeout;5000000"
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Keep only 1 decoded frame in OpenCV's internal ring-buffer.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.warning("Cannot open RTSP stream: %s", self.rtsp_url)
        else:
            logger.info(
                "Stream opened (TCP) — source %dx%d @ %.0f fps → output %dx%d",
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                cap.get(cv2.CAP_PROP_FPS),
                self._target_w,
                self._target_h,
            )
        return cap

    # ── Private: background reader loop ────────────────────────────────────
    def _update(self) -> None:
        cap = self._open_capture()

        while self.running:
            try:
                if not cap.isOpened():
                    logger.warning("Stream lost — reconnecting in 1 s …")
                    cap.release()
                    time.sleep(1)
                    cap = self._open_capture()
                    continue

                # Drain the OS TCP socket buffer aggressively by grabbing all available frames.
                grabbed_frame = False
                while True:
                    t_start = time.time()
                    ret = cap.grab()
                    if not ret:
                        break
                    grabbed_frame = True
                    # If grabbing took more than 3ms, we assume we blocked waiting for a new network packet,
                    # which means we have reached the live edge of the stream.
                    if time.time() - t_start > 0.003:
                        break

                if not grabbed_frame:
                    logger.warning("Stream read failed — reconnecting …")
                    cap.release()
                    time.sleep(0.5)
                    cap = self._open_capture()
                    continue

                # Only perform the expensive decode + resize step at 15 FPS
                now = time.time()
                if now - self._last_decode_time < (1.0 / 15.0):
                    continue
                self._last_decode_time = now

                # Decode the latest grabbed frame
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    continue

                # Pre-scale here once to the target stream resolution.
                # Every downstream consumer gets a small, cheap frame.
                if frame.shape[1] != self._target_w or frame.shape[0] != self._target_h:
                    frame = cv2.resize(
                        frame,
                        (self._target_w, self._target_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                # Atomic assignment — GIL guarantees consumers see a complete frame
                self._latest_frame = frame
                self.frame_ready_event.set()

            except Exception as exc:
                logger.error("Stream thread error: %s", exc)
                time.sleep(1)

        cap.release()
        logger.info("VideoStream stopped.")

    # ── Public API ──────────────────────────────────────────────────────────
    def read_latest(self):
        """Return the most-recent decoded frame, or None if not yet available. Never blocks."""
        return self._latest_frame

    def is_alive(self) -> bool:
        """Return True if the background reader thread is still running."""
        return self._thread.is_alive() and self.running

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self.running = False
        self._thread.join(timeout=3)
        logger.info("VideoStream reader thread joined.")

