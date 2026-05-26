"""
WebRTC video track for SafeVision.

``SafeVisionTrack.recv()`` grabs the latest annotated frame from the
pipeline and delivers it to the WebRTC peer as an ``av.VideoFrame``.

Optimisation: the BGR→YUV colour conversion performed by
``VideoFrame.from_ndarray`` is CPU-bound and blocks for 1–3 ms.  Running
it inside the asyncio event loop would stall other coroutines (e.g. the
``/offer`` signalling endpoint).  We push it to a small thread-pool
executor so the event loop stays responsive.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

# Small dedicated pool — 2 workers is enough since only one WebRTC track
# calls recv() at a time; the second worker handles overlap between frames.
_frame_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sv-webrtc")


def _convert_frame(raw_frame: np.ndarray) -> VideoFrame:
    """Blocking BGR→YUV conversion — runs in the thread pool."""
    return VideoFrame.from_ndarray(raw_frame, format="bgr24")


class SafeVisionTrack(VideoStreamTrack):
    """
    A WebRTC VideoStreamTrack that grabs the latest annotated raw frame
    from the SafeVision pipeline.
    """

    kind = "video"

    def __init__(self):
        super().__init__()
        self._start = time.time()
        self._timestamp = 0

        # Placeholder frame if pipeline hasn't produced one yet
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Camera Connecting or Offline", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self._placeholder = img

    async def recv(self):
        from app.api import get_latest_raw_frame

        # aiortc expects ~30 FPS timing
        pts, time_base = await self.next_timestamp()

        # Grab the latest frame from the pipeline
        raw_frame = get_latest_raw_frame()
        if raw_frame is None:
            raw_frame = self._placeholder

        # Offload the blocking BGR→YUV conversion to the thread pool so the
        # asyncio event loop is not stalled during the colour-space conversion.
        loop = asyncio.get_event_loop()
        new_frame = await loop.run_in_executor(_frame_executor, _convert_frame, raw_frame)
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame
