"""
Threaded RTSP video stream reader.

Runs a background thread that continuously reads frames from the RTSP
camera and keeps a bounded queue of the latest frames.  Consumers call
``read()`` to get the most recent frame.

Supports GPU-accelerated decode via GStreamer NVDEC (T4/desktop GPUs)
with automatic fallback to CPU FFmpeg decode.
"""

from app.config import settings
import logging
import queue
import shutil
import threading
import time
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
import cv2

logger = logging.getLogger("safevision")


def _build_gstreamer_pipeline(rtsp_url: str) -> str:
    """
    Build a GStreamer pipeline string for GPU-accelerated RTSP decode.

    Uses NVDEC (``nvh264dec`` / ``nvv4l2decoder``) for hardware H.264
    decoding on NVIDIA GPUs, eliminating CPU decode overhead.
    """
    return (
        f"rtspsrc location={rtsp_url} latency=100 protocols=tcp ! "
        f"rtph264depay ! h264parse ! "
        f"nvh264dec ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


def _gstreamer_available() -> bool:
    """Check if GStreamer backend is available in this OpenCV build."""
    try:
        # Check if OpenCV was built with GStreamer support
        build_info = cv2.getBuildInformation()
        return "GStreamer" in build_info and "YES" in build_info.split("GStreamer")[1].split("\n")[0]
    except Exception:
        return False


class VideoStream:
    """Non-blocking RTSP reader with automatic reconnection.

    Supports GPU-accelerated decode (GStreamer + NVDEC) with automatic
    fallback to CPU-based FFmpeg decode.
    """

    def __init__(
        self,
        rtsp_url: str | None = None,
        queue_size: int | None = None,
        use_gpu_decode: bool = True,
    ):
        self.rtsp_url = rtsp_url or settings.RTSP_URL
        self._queue_size = queue_size or settings.QUEUE_SIZE
        self._use_gpu_decode = use_gpu_decode and getattr(settings, 'USE_GPU_DECODE', True)

        # Try GPU decode first, fall back to CPU FFmpeg
        self.cap = self._open_capture()

        self.q: queue.Queue = queue.Queue(maxsize=self._queue_size)
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        logger.info("VideoStream started — %s", self.rtsp_url)

    def _open_capture(self) -> cv2.VideoCapture:
        """
        Open the RTSP stream, trying GPU decode first.

        Falls back to CPU FFmpeg if GStreamer/NVDEC is unavailable.
        """
        if self._use_gpu_decode and _gstreamer_available():
            try:
                pipeline = _build_gstreamer_pipeline(self.rtsp_url)
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    logger.info("GPU decode active (GStreamer + NVDEC)")
                    return cap
                else:
                    cap.release()
                    logger.warning("GStreamer pipeline failed to open — falling back to FFmpeg")
            except Exception as exc:
                logger.warning("GStreamer init failed: %s — falling back to FFmpeg", exc)

        # Fallback: CPU-based FFmpeg decode (original path)
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # ── Critical: kill OpenCV's internal frame buffer ──────────────────
        # By default OpenCV buffers 4–10 decoded frames internally, creating
        # 130–330 ms of hidden latency even before our queue code runs.
        # Setting BUFFERSIZE=1 means only the single most-recent decoded frame
        # is kept; older frames are discarded immediately.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.warning("Cannot open RTSP stream at startup: %s. Will keep retrying.", self.rtsp_url)
        else:
            logger.info("CPU decode active (FFmpeg)")
        return cap

    # ── Background reader ─────────────────────────────────────────────────
    def _update(self) -> None:
        fail_count = 0
        while self.running:
            try:
                if not self.cap.isOpened():
                    logger.warning("Stream lost — reconnecting …")
                    self.cap.release()
                    self.cap = self._open_capture()
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
