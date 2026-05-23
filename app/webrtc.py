import asyncio
import fractions
import time
import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

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

        # Convert OpenCV BGR numpy array to av.VideoFrame
        new_frame = VideoFrame.from_ndarray(raw_frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame
